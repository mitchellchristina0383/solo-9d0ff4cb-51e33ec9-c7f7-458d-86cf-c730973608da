// Browser-side SSE client for the failover gateway.
const form = document.getElementById("chat-form");
const contextInput = document.getElementById("context");
const promptInput = document.getElementById("prompt");
const maxTokensInput = document.getElementById("max-tokens");
const scenarioSelect = document.getElementById("scenario");
const runBtn = document.getElementById("run-btn");
const preheatBtn = document.getElementById("preheat-btn");
const cacheStatusEl = document.getElementById("cache-status");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const eventsEl = document.getElementById("events");
const counterEl = document.getElementById("counter");
const sharesEl = document.getElementById("shares");
const cacheHitsEl = document.getElementById("cache-hits");
const verificationEl = document.getElementById("verification");

let expected = null;

// The complete input: common context + this question. The gateway joins them
// the same way; /api/expected must be queried with the joined text.
function fullPrompt() {
  return contextInput.value + promptInput.value;
}

async function loadScenarios() {
  const resp = await fetch("/api/scenarios");
  const data = await resp.json();
  scenarioSelect.innerHTML = "";
  for (const s of data.scenarios) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.id;
    scenarioSelect.appendChild(opt);
  }
}

async function loadCacheStatus() {
  try {
    const resp = await fetch("/api/cache");
    const data = await resp.json();
    const parts = Object.entries(data.nodes).map(([node, info]) => {
      if (!info || info.ok === false) return `${node}: 不可达`;
      if (!info.cached) return `${node}: 空`;
      const st = info.stats || {};
      return `${node}: 已缓存 ${info.context_len} 字符 (state=${info.state_prefix}…, ` +
        `命中${st.hits ?? 0}/重算${st.misses ?? 0})`;
    });
    cacheStatusEl.textContent = `节点缓存：${parts.join("；")}`;
  } catch (err) {
    cacheStatusEl.textContent = `缓存状态查询失败：${err.message}`;
  }
}

async function preheat() {
  const context = contextInput.value;
  if (!context) {
    addEvent("warn", "⛔ 请先填写公共上下文再预热");
    return;
  }
  preheatBtn.disabled = true;
  try {
    const resp = await fetch("/api/preheat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context }),
    });
    const data = await resp.json();
    for (const [node, r] of Object.entries(data.results)) {
      if (r.ok) {
        const extra = r.replaced ? "（替换了上一段上下文）" : r.refreshed ? "（刷新同一段上下文）" : "";
        addEvent("ok", `🔥 ${node} 预热完成：${r.context_len} 字符，state=${r.state_prefix}…${extra}`);
      } else {
        addEvent("warn", `⛔ ${node} 预热失败：${r.error}`);
      }
    }
    await loadCacheStatus();
  } catch (err) {
    addEvent("warn", `⛔ 预热请求失败：${err.message}`);
  } finally {
    preheatBtn.disabled = false;
  }
}

function setStatus(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

function addEvent(kind, html) {
  const div = document.createElement("div");
  div.className = `event-line ${kind}`;
  div.innerHTML = html;
  eventsEl.appendChild(div);
}

// Mirror of the gateway's append-only guard: refuse duplicate / reordered
// indices so a late packet could never mutate what the user already sees.
let lastIndex = -1;
function packetOrderGuard(index) {
  if (index !== lastIndex + 1) {
    addEvent("warn", `⚠️ 忽略乱序/迟到词元 index=${index}（下一个应为 ${lastIndex + 1}）`);
    throw new Error("out-of-order token");
  }
  lastIndex = index;
}

function appendToken(obj) {
  const placeholder = outputEl.querySelector(".placeholder");
  if (placeholder) placeholder.remove();
  packetOrderGuard(obj.index);
  // Append-only: never overwrite previously displayed tokens.
  const chip = document.createElement("span");
  chip.className = `token ${obj.node}`;
  chip.dataset.index = obj.index;
  chip.title = `index=${obj.index} id=${obj.id} 来源=${obj.node}`;
  chip.textContent = obj.text;
  outputEl.appendChild(chip);
  counterEl.textContent = `已采纳词元：${obj.index + 1}`;
}

function renderShares(nodes) {
  if (!nodes) return;
  sharesEl.textContent = `各节点词元占比：${Object.entries(nodes)
    .map(([node, n]) => `${node}=${n}`).join("，")}`;
}

// Per-participating-node cache chips, driven by the gateway's `cache` events
// (one per node that actually joined the generation).
const cacheChips = new Map();
function renderCacheChip(obj) {
  let chip = cacheChips.get(obj.node);
  if (!chip) {
    chip = document.createElement("span");
    cacheChips.set(obj.node, chip);
    cacheHitsEl.appendChild(chip);
  }
  chip.className = `cache-chip ${obj.cache_hit ? "hit" : "miss"}`;
  chip.textContent = `${obj.node}：${obj.cache_hit ? "缓存命中" : "未命中·按原文重算"}`;
  chip.title = `上下文 ${obj.context_len} 字符，state=${obj.context_state || "—"}`;
}

async function verifyAgainstHealthy(tokens, incomplete) {
  if (!expected) {
    const resp = await fetch(
      `/api/expected?prompt=${encodeURIComponent(fullPrompt())}&max_tokens=${maxTokensInput.value}`
    );
    expected = await resp.json();
  }
  const got = tokens.map((t) => t.id);
  const n = got.length;
  if (incomplete) {
    const prefixOk = JSON.stringify(got) === JSON.stringify(expected.ids.slice(0, n));
    verificationEl.className = `verification ${prefixOk ? "pass" : "fail"}`;
    verificationEl.innerHTML = prefixOk
      ? `✅ 未完成：保留的 ${n} 个词元与无故障基准的同位置前缀完全一致（已显示内容未被篡改）。`
      : "❌ 保留前缀与基准不一致！";
    return;
  }
  const fullOk = JSON.stringify(got) === JSON.stringify(expected.ids);
  verificationEl.className = `verification ${fullOk ? "pass" : "fail"}`;
  verificationEl.innerHTML = fullOk
    ? "✅ 接续成功：整条回复（首节点前缀 + 备用节点续写）与无故障 healthy 场景的词元序列完全一致。"
    : "❌ failover 结果与 healthy 不一致！";
}

// Minimal SSE parser over fetch() POST (EventSource cannot POST).
async function consumeSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = null;
  const dataLines = [];

  const dispatch = () => {
    if (event !== null || dataLines.length) {
      onEvent(event || "message", dataLines.join("\n"));
    }
    event = null;
    dataLines.length = 0;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, sep).replace(/\r$/, "");
      buffer = buffer.slice(sep + 1);
      if (line === "") {
        dispatch();
      } else if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).replace(/^ /, ""));
      }
    }
  }
  dispatch();
}

async function run() {
  runBtn.disabled = true;
  setStatus("running", "生成中…");
  outputEl.innerHTML = '<span class="placeholder">（回复将在这里逐词元出现）</span>';
  eventsEl.innerHTML = "";
  sharesEl.textContent = "";
  cacheHitsEl.innerHTML = "";
  cacheChips.clear();
  verificationEl.className = "verification hidden";
  verificationEl.innerHTML = "";
  counterEl.textContent = "已采纳词元：0";
  lastIndex = -1;
  expected = null;
  const tokens = [];
  let incomplete = false;

  const req = {
    context: contextInput.value,
    prompt: promptInput.value,
    max_tokens: parseInt(maxTokensInput.value, 10),
    scenario: scenarioSelect.value,
  };

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`网关返回 HTTP ${resp.status}`);
    }

    await consumeSSE(resp, (event, data) => {
      const obj = JSON.parse(data);
      switch (event) {
        case "meta":
          addEvent("", `请求 ${obj.request_id.slice(0, 8)}… 场景=<b>${obj.scenario}</b>，词元上限=${obj.max_tokens}` +
            (obj.context_len ? `，公共上下文=${obj.context_len} 字符` : "（无公共上下文）"));
          break;
        case "cache":
          renderCacheChip(obj);
          addEvent("", `🗄 ${obj.node}：${obj.cache_hit ? "上下文缓存命中，复用预热结果" : "缓存未命中，按原文重算上下文"}`);
          break;
        case "token":
          appendToken(obj);
          tokens.push(obj);
          break;
        case "fallback":
          addEvent("warn",
            `⏱ 首节点 <b>${obj.from_node}</b> ${obj.reason === "read_timeout" ? "读超时" : "出错"}（已采纳 ${obj.accepted_count} 个词元），切换到 <b>${obj.to_node}</b> 续写…`);
          break;
        case "incomplete":
          incomplete = true;
          addEvent("warn", `⛔ 标记为未完成：${obj.reason}（保留 ${obj.accepted_count} 个已显示词元）`);
          break;
        case "done":
          renderShares(obj.nodes);
          addEvent("ok", `✔ 完成：共 ${obj.token_count} 个词元`);
          break;
        default:
          addEvent("", JSON.stringify(obj));
      }
    });

    setStatus(incomplete ? "incomplete" : "done", incomplete ? "未完成" : "完成");
    await verifyAgainstHealthy(tokens, incomplete);
    await loadCacheStatus();
  } catch (err) {
    setStatus("incomplete", "请求失败");
    addEvent("warn", `⛔ ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  run();
});
preheatBtn.addEventListener("click", preheat);
loadScenarios();
loadCacheStatus();
