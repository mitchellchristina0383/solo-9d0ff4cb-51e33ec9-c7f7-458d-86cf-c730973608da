"""Real-HTTP trial run of every scenario (no Docker required).

Spins up the actual uvicorn servers for node-a, node-b and the gateway on
localhost ports, then drives the SSE API exactly like the browser does and
prints a full report: accepted tokens per node, failover events, the late
packets the abandoned primary emitted afterwards, and the equivalence check
against the healthy baseline.

The second half exercises the shared-context preheat / prefix-cache feature:
preheat both nodes, generate with cache hits, fail over with both nodes
hitting, evict the context with a different preheat and confirm the reply is
still token-identical (recomputed from the original text), and finally
interleave concurrent generations with a mid-flight preheat.

Usage:  python run_demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time

import httpx
import uvicorn

from common.deterministic import expected_token_ids
from gateway.app import GatewayConfig, create_app as create_gateway
from node.app import NodeConfig, create_app as create_node

PROMPT = "请写一段关于秋天的短文"
MAX_TOKENS = 12
PORT_A, PORT_B, PORT_G = 8901, 8902, 8908
READ_TIMEOUT = 0.5
EMIT_INTERVAL = 0.05

CONTEXT = "《项目纪要》第三季度网关改造完成，两个推理节点互为主备，词元上限统一由网关管理。"
CONTEXT_B = "另一份共享资料：演示环境的所有节点词表版本均为 tok-v1。"
QUESTION = "请根据资料写一句总结"


def serve(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


async def consume(client: httpx.AsyncClient, scenario: str, context: str = "",
                  prompt: str = PROMPT, max_tokens: int = MAX_TOKENS):
    events: list[tuple[str, dict]] = []
    async with client.stream(
        "POST", "http://127.0.0.1:8908/api/chat",
        json={"context": context, "prompt": prompt,
              "max_tokens": max_tokens, "scenario": scenario},
    ) as resp:
        event = None
        data: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event is not None or data:
                    events.append((event or "message", json.loads("\n".join(data))))
                event, data = None, []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
    return events


def report(scenario: str, events, healthy_ids, max_tokens: int = MAX_TOKENS):
    print(f"\n{'=' * 78}\n场景: {scenario}\n{'=' * 78}")
    tokens = [e for n, e in events if n == "token"]
    for name, e in events:
        if name == "meta":
            ctx = f"，公共上下文={e['context_len']} 字符" if e.get("context_len") else ""
            print(f"[meta] request={e['request_id'][:8]}… 上限={e['max_tokens']}{ctx}")
        elif name == "cache":
            mark = "命中，复用预热结果" if e["cache_hit"] else "未命中，按原文重算"
            print(f"[cache] {e['node']}：{mark} (state={e.get('context_state') or '—'})")
        elif name == "fallback":
            print(f"[fallback] {e['reason']}：{e['from_node']} 已采纳 {e['accepted_count']} "
                  f"个词元后切换到 {e['to_node']}")
        elif name == "incomplete":
            print(f"[incomplete] reason={e['reason']} 保留 {e['accepted_count']} 个词元")
            if "primary_identity" in e:
                print(f"             primary={e['primary_identity']}")
                print(f"             backup ={e['backup_identity']}")
    by_node: dict[str, list[str]] = {}
    for t in tokens:
        by_node.setdefault(t["node"], []).append(t["text"])
    for node, texts in by_node.items():
        print(f"[tokens:{node}] ({len(texts)}) {' '.join(texts)}")
    ids = [t["id"] for t in tokens]
    final = "done" in {n for n, _ in events}
    if final:
        ok = ids == healthy_ids
        print(f"[核对] 完整回复词元数={len(ids)}/{max_tokens}，与基准一致: {ok}")
        assert ok, "result diverged from the direct-computation baseline"
    else:
        n = len(ids)
        ok = ids == healthy_ids[:n]
        print(f"[核对] 未完成，保留前缀 {n}/{max_tokens} 个词元，与基准同位置前缀一致: {ok}")
        assert ok
    return tokens


async def preheat_demo(client: httpx.AsyncClient):
    print(f"\n{'#' * 78}\n# 公共上下文预热 / 前缀缓存演示\n{'#' * 78}")

    print("\n-- 1. 预热同一段上下文到两台节点（各节点只留最近一次预热）--")
    r = await client.post("http://127.0.0.1:8908/api/preheat", json={"context": CONTEXT})
    data = r.json()
    for node, res in sorted(data["results"].items()):
        print(f"   {node}: ok={res['ok']} {res['context_len']} 字符 "
              f"state={res['state_prefix']}… replaced={res['replaced']}")
    assert data["ok"]

    baseline = expected_token_ids(CONTEXT + QUESTION, 0, MAX_TOKENS)
    print(f"-- 2. 带着该上下文提问（healthy）：期望 node-a 缓存命中，词元与直接计算一致 --")
    report("preheat_hit (healthy)", await consume(client, "healthy", CONTEXT, QUESTION), baseline)

    print("-- 3. 同一上下文 + 切换场景：首节点超时后备用节点续写，两台都应命中 --")
    report("preheat_hit (failover_success)",
           await consume(client, "failover_success", CONTEXT, QUESTION), baseline)

    print("-- 4. 换一段上下文预热（顶掉旧缓存），再用原上下文提问：应未命中重算，词元仍逐位一致 --")
    r = await client.post("http://127.0.0.1:8908/api/preheat", json={"context": CONTEXT_B})
    assert r.json()["ok"]
    report("evicted_then_miss (healthy)", await consume(client, "healthy", CONTEXT, QUESTION), baseline)

    print("-- 5. 并发交错：两路生成 + 中途又一次预热，所有回复仍与各自直接计算一致 --")
    q1, q2, q3 = "并发问题一", "并发问题二", "并发问题三"

    async def delayed_preheat():
        await asyncio.sleep(0.08)
        await client.post("http://127.0.0.1:8908/api/preheat", json={"context": CONTEXT})

    (e1, e2, _, e3) = await asyncio.gather(
        consume(client, "healthy", CONTEXT, q1),
        consume(client, "healthy", CONTEXT_B, q2),
        delayed_preheat(),
        consume(client, "healthy", CONTEXT, q3),
    )
    for events, ctx, q in ((e1, CONTEXT, q1), (e2, CONTEXT_B, q2), (e3, CONTEXT, q3)):
        ids = [e["id"] for n, e in events if n == "token"]
        ok = ids == expected_token_ids(ctx + q, 0, MAX_TOKENS)
        hits = {e["node"]: e["cache_hit"] for n, e in events if n == "cache"}
        print(f"   并发 {q!r}（上下文 {len(ctx)} 字符）：词元一致={ok}，命中={hits}")
        assert ok

    print("-- 6. 网关聚合的节点缓存状态 --")
    status = (await client.get("http://127.0.0.1:8908/api/cache")).json()
    for node, info in sorted(status["nodes"].items()):
        st = info.get("stats", {})
        print(f"   {node}: cached={info.get('cached')} 上下文={info.get('context_len')} 字符 "
              f"state={info.get('state_prefix')}… 预热{st.get('preheats')}次 "
              f"命中{st.get('hits')}次 重算{st.get('misses')}次")


async def main():
    node_a = create_node(NodeConfig("node-a", emit_interval=EMIT_INTERVAL))
    node_b = create_node(NodeConfig("node-b", emit_interval=EMIT_INTERVAL))
    gateway = create_gateway(GatewayConfig(
        node_a_url=f"http://127.0.0.1:{PORT_A}",
        node_b_url=f"http://127.0.0.1:{PORT_B}",
        read_timeout=READ_TIMEOUT,
    ))
    servers = [serve(node_a, PORT_A), serve(node_b, PORT_B), serve(gateway, PORT_G)]
    # Wait for readiness.
    async with httpx.AsyncClient() as client:
        for port in (PORT_A, PORT_B, PORT_G):
            for _ in range(50):
                try:
                    r = await client.get(f"http://127.0.0.1:{port}/info" if port != PORT_G
                                         else f"http://127.0.0.1:{port}/api/scenarios")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.1)

    healthy_ids = expected_token_ids(PROMPT, 0, MAX_TOKENS)
    print(f"确定性基准词元 id 序列: {healthy_ids}")
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            healthy = report("healthy", await consume(client, "healthy"), healthy_ids)
            report("failover_success", await consume(client, "failover_success"), healthy_ids)
            report("failover_backup_error", await consume(client, "failover_backup_error"), healthy_ids)
            report("version_mismatch", await consume(client, "version_mismatch"), healthy_ids)

            # Prove the abandoned primary really sent late packets afterwards.
            await asyncio.sleep(2.2)
            late = (await client.get(f"http://127.0.0.1:{PORT_A}/debug/late")).json()
            flat = [p for vals in late.values() for p in vals]
            print(f"\n[迟到回包] node-a 在被放弃后仍产生 {len(flat)} 个迟到词元 "
                  f"(indices={sorted({p['index'] for p in flat})})，网关均未采纳。")
            assert flat

            await preheat_demo(client)
    finally:
        for s in servers:
            s.should_exit = True
    print(f"\n全部场景试跑通过，耗时 {time.time() - started:.1f}s。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\n试跑失败: {exc}", file=sys.stderr)
        sys.exit(1)
