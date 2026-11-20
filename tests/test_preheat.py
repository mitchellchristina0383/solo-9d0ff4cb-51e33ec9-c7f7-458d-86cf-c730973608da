"""Behavior tests for the shared-context preheat / prefix-cache feature.

Contract under test:

* The page submits the common context and the question separately; the
  complete input (context + question) is authoritative for generation and
  for any backup continuation.
* POST /api/preheat warms BOTH mock nodes; each node keeps only its most
  recent successful preheat (single slot, replacement on new preheat,
  in-memory only so a restart loses it).
* A generate whose context matches the cached entry reuses the precomputed
  context state (hit); otherwise the node recomputes from the original text
  (miss). Hit or miss, the token stream is identical to direct computation —
  including under concurrency and interleaved preheats of other contexts.
* The gateway forwards one `cache` event per node that actually participates
  and summarizes hits in done/incomplete.

Tests that need a pristine cache slot boot their own throwaway node servers
(FreshNode), so they stay independent of the session-shared harness state.

Run from the repository root:  python -m pytest -v
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import uvicorn

from common.deterministic import expected_token_ids
from node.app import NodeConfig, create_app as create_node

CONTEXT = "《项目纪要》第三季度网关改造完成，两个推理节点互为主备。"
CONTEXT_B = "另一份完全不同的共享资料，用于顶掉前一次预热。"
QUESTION = "请根据资料写一句总结"
MAX_TOKENS = 10


class FreshNode:
    """A throwaway node server on a free port (its own empty cache slot)."""

    def __init__(self, node_id: str = "node-fresh"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(
            create_node(NodeConfig(node_id, emit_interval=0.01)),
            host="127.0.0.1", port=self.port, log_level="error",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> "FreshNode":
        self.thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("fresh node failed to start")

    def stop(self):
        self.server.should_exit = True

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=10.0)


async def collect_sse(client: httpx.AsyncClient, payload: dict) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        event = None
        data_lines: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event is not None or data_lines:
                    events.append((event or "message", json.loads("\n".join(data_lines))))
                event, data_lines = None, []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if event is not None or data_lines:
            events.append((event or "message", json.loads("\n".join(data_lines))))
    return events


def chat_payload(context: str, question: str, scenario: str = "healthy",
                 max_tokens: int = MAX_TOKENS) -> dict:
    return {"context": context, "prompt": question,
            "max_tokens": max_tokens, "scenario": scenario}


def tokens_of(events):
    return [(e["index"], e["id"], e["text"], e["node"]) for name, e in events if name == "token"]


def cache_events_of(events):
    return [e for name, e in events if name == "cache"]


def done_of(events):
    return next(e for n, e in events if n == "done")


async def preheat(client: httpx.AsyncClient, context: str) -> dict:
    resp = await client.post("/api/preheat", json={"context": context})
    assert resp.status_code == 200
    return resp.json()


async def node_generate_ids(client: httpx.AsyncClient, prompt: str, context: str,
                            max_tokens: int) -> tuple[list[int], bool]:
    """Direct node-level /generate; returns (token ids, cache_hit)."""
    ids: list[int] = []
    hit = None
    async with client.stream("POST", "/generate", json={
        "request_id": "r", "prompt": prompt, "context": context,
        "start": 0, "max_tokens": max_tokens,
    }) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                data = json.loads(line[5:].lstrip())
                if "cache_hit" in data:
                    hit = data["cache_hit"]
                if "id" in data:
                    ids.append(data["id"])
    return ids, hit


# --- async scenarios ---------------------------------------------------------

async def _preheat_fans_out_to_both_nodes(gateway_client):
    data = await preheat(gateway_client, CONTEXT)
    assert data["ok"] is True
    assert set(data["results"]) == {"node-a", "node-b"}
    for node in ("node-a", "node-b"):
        r = data["results"][node]
        assert r["ok"] and r["cached"] is True
        assert r["context_len"] == len(CONTEXT)
    # A second preheat of a different context replaces the first on both nodes.
    data2 = await preheat(gateway_client, CONTEXT_B)
    for node in ("node-a", "node-b"):
        assert data2["results"][node]["replaced"] is True
    # Gateway cache view agrees: both nodes now hold CONTEXT_B only.
    status = (await gateway_client.get("/api/cache")).json()
    for node in ("node-a", "node-b"):
        info = status["nodes"][node]
        assert info["cached"] is True
        assert info["context_len"] == len(CONTEXT_B)
        assert info["context_preview"] in CONTEXT_B


async def _single_slot_keeps_only_latest_preheat():
    node = FreshNode().start()
    try:
        async with node.client() as client:
            r1 = (await client.post("/preheat", json={"context": CONTEXT})).json()
            assert r1["replaced"] is False and r1["refreshed"] is False
            r2 = (await client.post("/preheat", json={"context": CONTEXT})).json()
            assert r2["refreshed"] is True and r2["replaced"] is False
            r3 = (await client.post("/preheat", json={"context": CONTEXT_B})).json()
            assert r3["replaced"] is True
            status = (await client.get("/cache")).json()
            # Only the most recent preheat survives.
            assert status["cached"] is True
            assert status["context_len"] == len(CONTEXT_B)
            assert status["context_preview"] in CONTEXT_B
            assert status["stats"]["preheats"] == 3
    finally:
        node.stop()


async def _hit_reuses_preheat_and_matches_direct(gateway_client):
    await preheat(gateway_client, CONTEXT)
    events = await collect_sse(gateway_client, chat_payload(CONTEXT, QUESTION))
    tokens = tokens_of(events)
    assert len(tokens) == MAX_TOKENS
    # Token-identical to direct computation over the complete input.
    assert [t[1] for t in tokens] == expected_token_ids(CONTEXT + QUESTION, 0, MAX_TOKENS)
    # Only the primary participated; it reports a hit.
    caches = cache_events_of(events)
    assert [(c["node"], c["cache_hit"]) for c in caches] == [("node-a", True)]
    assert caches[0]["context_len"] == len(CONTEXT)
    assert done_of(events)["cache"] == {"node-a": True}


async def _miss_after_replacement_still_identical(gateway_client):
    await preheat(gateway_client, CONTEXT)
    await preheat(gateway_client, CONTEXT_B)  # evicts CONTEXT on both nodes
    events = await collect_sse(gateway_client, chat_payload(CONTEXT, QUESTION))
    tokens = tokens_of(events)
    assert [t[1] for t in tokens] == expected_token_ids(CONTEXT + QUESTION, 0, MAX_TOKENS)
    caches = cache_events_of(events)
    assert [(c["node"], c["cache_hit"]) for c in caches] == [("node-a", False)]
    assert done_of(events)["cache"] == {"node-a": False}


async def _generate_miss_does_not_fill_cache():
    node = FreshNode().start()
    try:
        async with node.client() as client:
            ids, hit = await node_generate_ids(client, CONTEXT + QUESTION, CONTEXT, 5)
            assert hit is False  # nothing preheated: recompute from text
            assert ids == expected_token_ids(CONTEXT + QUESTION, 0, 5)
            status = (await client.get("/cache")).json()
            # The generate-time recompute was discarded, not cached.
            assert status["cached"] is False
            assert status["stats"]["preheats"] == 0
            assert status["stats"]["misses"] == 1
    finally:
        node.stop()


async def _failover_with_preheat_both_nodes_hit(gateway_client):
    await preheat(gateway_client, CONTEXT)
    events = await collect_sse(gateway_client, chat_payload(CONTEXT, QUESTION, "failover_success"))
    names = [n for n, _ in events]
    assert "fallback" in names
    tokens = tokens_of(events)
    assert len(tokens) == MAX_TOKENS
    assert [t[3] for t in tokens] == ["node-a"] * 4 + ["node-b"] * (MAX_TOKENS - 4)
    # Continuation over the same complete input reproduces the healthy stream.
    assert [t[1] for t in tokens] == expected_token_ids(CONTEXT + QUESTION, 0, MAX_TOKENS)
    # Both participating nodes reused their preheated context state.
    caches = cache_events_of(events)
    assert [(c["node"], c["cache_hit"]) for c in caches] == [("node-a", True), ("node-b", True)]
    done = done_of(events)
    assert done["cache"] == {"node-a": True, "node-b": True}
    assert done["identical_to_healthy"] is True


async def _failover_after_replacement_recomputes_and_stays_identical(gateway_client):
    await preheat(gateway_client, CONTEXT_B)  # both nodes cache CONTEXT_B
    events = await collect_sse(gateway_client, chat_payload(CONTEXT, QUESTION, "failover_success"))
    tokens = tokens_of(events)
    assert [t[1] for t in tokens] == expected_token_ids(CONTEXT + QUESTION, 0, MAX_TOKENS)
    caches = cache_events_of(events)
    assert [(c["node"], c["cache_hit"]) for c in caches] == [("node-a", False), ("node-b", False)]
    assert done_of(events)["cache"] == {"node-a": False, "node-b": False}


async def _concurrent_interleave_stays_token_identical(gateway_client):
    await preheat(gateway_client, CONTEXT)

    async def generate(context: str, question: str):
        events = await collect_sse(gateway_client, chat_payload(context, question))
        return tokens_of(events), cache_events_of(events), done_of(events)

    async def delayed_preheat(context: str):
        await asyncio.sleep(0.05)  # land mid-stream, replacing the slot
        await preheat(gateway_client, context)

    q1, q2, q3 = "并发问题一", "并发问题二", "并发问题三"
    (r1, r2, _, r3) = await asyncio.gather(
        generate(CONTEXT, q1),
        generate(CONTEXT_B, q2),
        delayed_preheat(CONTEXT_B),  # evicts CONTEXT while streams run
        generate(CONTEXT, q3),
    )
    # Whatever the cache did, every reply equals direct computation.
    for (tokens, caches, done), context, question in (
        (r1, CONTEXT, q1), (r2, CONTEXT_B, q2), (r3, CONTEXT, q3),
    ):
        assert [t[1] for t in tokens] == expected_token_ids(context + question, 0, MAX_TOKENS)
        assert len(tokens) == MAX_TOKENS
        for c in caches:
            assert c["cache_hit"] in (True, False)
            assert c["context_len"] == len(context)
        assert all(isinstance(v, bool) for v in done["cache"].values())


async def _restart_loses_cache():
    """A fresh node process (new app instance = restart) starts with an empty
    slot and recomputes from the original text; tokens stay identical."""
    # Boot 1: preheat, confirm hit.
    node1 = FreshNode("node-boot1").start()
    async with node1.client() as client:
        r = await client.post("/preheat", json={"context": CONTEXT})
        assert r.status_code == 200
        ids, hit = await node_generate_ids(client, CONTEXT + QUESTION, CONTEXT, 5)
        assert hit is True
        assert ids == expected_token_ids(CONTEXT + QUESTION, 0, 5)
    node1.stop()

    # Boot 2 ("restart"): brand-new app instance -> empty slot.
    node2 = FreshNode("node-boot2").start()
    try:
        async with node2.client() as client:
            status = (await client.get("/cache")).json()
            assert status["cached"] is False
            ids, hit = await node_generate_ids(client, CONTEXT + QUESTION, CONTEXT, 5)
            assert hit is False  # recomputed from the original text
            assert ids == expected_token_ids(CONTEXT + QUESTION, 0, 5)
    finally:
        node2.stop()


async def _context_not_a_prefix_never_hits(registry):
    """Defensive semantics: the submitted prompt is authoritative; a context
    that is not actually its prefix can never be a cache hit."""
    client = registry.client(registry.node_a_url)
    async with client:
        await client.post("/preheat", json={"context": CONTEXT})
        ids, hit = await node_generate_ids(client, "完全不同的开头" + QUESTION, CONTEXT, 4)
        assert hit is False
        assert ids == expected_token_ids("完全不同的开头" + QUESTION, 0, 4)


# --- sync wrappers -----------------------------------------------------------

def test_preheat_fans_out_to_both_nodes(gateway_client):
    asyncio.run(_preheat_fans_out_to_both_nodes(gateway_client))


def test_single_slot_keeps_only_latest_preheat():
    asyncio.run(_single_slot_keeps_only_latest_preheat())


def test_hit_reuses_preheat_and_matches_direct(gateway_client):
    asyncio.run(_hit_reuses_preheat_and_matches_direct(gateway_client))


def test_miss_after_replacement_still_identical(gateway_client):
    asyncio.run(_miss_after_replacement_still_identical(gateway_client))


def test_generate_miss_does_not_fill_cache():
    asyncio.run(_generate_miss_does_not_fill_cache())


def test_failover_with_preheat_both_nodes_hit(gateway_client):
    asyncio.run(_failover_with_preheat_both_nodes_hit(gateway_client))


def test_failover_after_replacement_recomputes_and_stays_identical(gateway_client):
    asyncio.run(_failover_after_replacement_recomputes_and_stays_identical(gateway_client))


def test_concurrent_interleave_stays_token_identical(gateway_client):
    asyncio.run(_concurrent_interleave_stays_token_identical(gateway_client))


def test_restart_loses_cache_and_recomputes():
    asyncio.run(_restart_loses_cache())


def test_context_not_a_prefix_never_hits(registry):
    asyncio.run(_context_not_a_prefix_never_hits(registry))
