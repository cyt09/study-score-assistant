/* 前端逻辑：只做展示与交互，所有计算与校验都在后端完成。 */

const state = {
  sessionId: localStorage.getItem("ssa_session") || null,
  sending: false,
  /* 搜索状态：active 为真时左侧列表展示的是搜索结果而不是会话列表 */
  search: { active: false, raw: "", mode: "keyword", timer: null },
};

const el = {
  app: document.getElementById("app"),
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("empty-state"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  composer: document.getElementById("composer"),
  statusDot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  history: document.getElementById("history"),
  historyTitle: document.getElementById("history-title"),
  searchInput: document.getElementById("search-input"),
  searchClear: document.getElementById("search-clear"),
  searchHint: document.getElementById("search-hint"),
  sessionId: document.getElementById("session-id"),
  sessionCount: document.getElementById("session-count"),
  refresh: document.getElementById("refresh-history"),
  clearHistory: document.getElementById("clear-history"),
  reset: document.getElementById("reset-session"),
  scrollBottom: document.getElementById("scroll-bottom"),
  sidebarOpen: document.getElementById("sidebar-open"),
  sidebarClose: document.getElementById("sidebar-close"),
  /* 数据看板 */
  kpiSessions: document.getElementById("kpi-sessions"),
  kpiTurns: document.getElementById("kpi-turns"),
  kpiSlope: document.getElementById("kpi-slope"),
  kpiSlopeFoot: document.getElementById("kpi-slope-foot"),
  kpiPrediction: document.getElementById("kpi-prediction"),
  kpiPredictionFoot: document.getElementById("kpi-prediction-foot"),
  distTotal: document.getElementById("dist-total"),
  barTrend: document.getElementById("bar-trend"),
  barRetrieval: document.getElementById("bar-retrieval"),
  valTrend: document.getElementById("val-trend"),
  valRetrieval: document.getElementById("val-retrieval"),
  sparkline: document.getElementById("sparkline"),
  sparkEmpty: document.getElementById("spark-empty"),
  sparkCaption: document.getElementById("spark-caption"),
  insightUpdated: document.getElementById("insight-updated"),
};

function setStatus(text, kind = "") {
  el.statusText.textContent = text;
  el.statusDot.className = "dot" + (kind ? " " + kind : "");
}

function syncSessionUI() {
  el.sessionId.textContent = state.sessionId ? state.sessionId.slice(0, 12) : "—";
  el.sessionId.title = state.sessionId || "";
}

function scrollToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
  updateScrollButton();
}

function hideEmptyState() {
  if (el.emptyState && el.emptyState.parentNode) el.emptyState.remove();
}

/* ---------------- 消息渲染 ---------------- */

function appendUserMessage(text) {
  hideEmptyState();
  const wrapper = document.createElement("div");
  wrapper.className = "msg user";
  wrapper.innerHTML = '<div class="avatar"></div><div class="bubble"></div>';
  wrapper.querySelector(".bubble").textContent = text;
  el.messages.appendChild(wrapper);
  scrollToBottom();
}

function appendLoading() {
  hideEmptyState();
  const wrapper = document.createElement("div");
  wrapper.className = "msg";
  wrapper.innerHTML = `
    <div class="avatar"></div>
    <div class="bubble">
      <div class="loading"><span class="spinner"></span><span class="loading-text">模型正在理解问题…</span></div>
    </div>`;
  el.messages.appendChild(wrapper);
  scrollToBottom();

  const phrases = ["模型正在理解问题…", "正在决定是否调用工具…", "正在执行计算与检索…", "正在校验输出字段…"];
  let index = 0;
  const node = wrapper.querySelector(".loading-text");
  const timer = setInterval(() => {
    index = (index + 1) % phrases.length;
    if (node) node.textContent = phrases[index];
  }, 1600);

  return {
    wrapper,
    stop: () => clearInterval(timer),
    // 后端 SSE 推进度时接管文案：停掉轮播，显示真实阶段
    setPhase: (text) => {
      clearInterval(timer);
      if (node && text) node.textContent = text;
    },
  };
}

function statBlock(label, value) {
  const div = document.createElement("div");
  div.className = "stat";
  div.innerHTML = `${label}<b></b>`;
  div.querySelector("b").textContent = value;
  return div;
}

function renderAnswer(payload) {
  const answer = payload.reply;
  const card = document.createElement("div");
  card.className = "answer-card";

  const text = document.createElement("p");
  text.className = "answer-text";
  text.textContent = answer.answer;
  card.appendChild(text);

  // 需要用户补充/修正时，直接给出可编辑的输入框和快捷按钮，而不是只说"请修改"
  if (payload.fix) card.appendChild(buildFixCard(payload.fix));

  const stats = document.createElement("div");
  stats.className = "stats";
  stats.appendChild(statBlock("趋势", answer.trend));
  if (answer.slope !== null && answer.slope !== undefined) {
    stats.appendChild(statBlock("斜率(分/次)", formatNum(answer.slope)));
  }
  if (answer.intercept !== null && answer.intercept !== undefined) {
    stats.appendChild(statBlock("截距", formatNum(answer.intercept)));
  }
  if (answer.prediction !== null && answer.prediction !== undefined) {
    stats.appendChild(statBlock("预测值", formatNum(answer.prediction)));
  }
  // 预测区间由程序计算后直接下发，不经过模型转述
  if (payload.prediction_interval) {
    const band = payload.prediction_interval;
    stats.appendChild(
      statBlock(
        `预测区间 ${Math.round((band.level || 0.95) * 100)}%`,
        `${formatNum(band.low)}–${formatNum(band.high)}`,
      ),
    );
  }
  card.appendChild(stats);

  if (payload.chart_url) {
    const img = document.createElement("img");
    img.className = "chart";
    img.src = payload.chart_url;
    img.alt = "成绩趋势拟合图";
    card.appendChild(img);
  }

  if (answer.prediction_note) {
    const note = document.createElement("p");
    note.className = "muted small";
    note.textContent = "预测说明：" + answer.prediction_note;
    card.appendChild(note);
  }

  const limitTitle = document.createElement("div");
  limitTitle.className = "section-title";
  limitTitle.textContent = "限制条件";
  card.appendChild(limitTitle);
  const list = document.createElement("ul");
  list.className = "limits";
  answer.limitations.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
  card.appendChild(list);

  if (answer.sources && answer.sources.length) {
    const srcTitle = document.createElement("div");
    srcTitle.className = "section-title";
    srcTitle.textContent = "资料出处";
    card.appendChild(srcTitle);
    const box = document.createElement("div");
    box.className = "sources";
    answer.sources.forEach((src) => {
      const item = document.createElement("div");
      item.className = "source";
      const head = document.createElement("div");
      head.className = "src-head";
      head.textContent = `${src.title} · ${src.chunk_id}`;
      const body = document.createElement("div");
      body.textContent = src.snippet;
      item.append(head, body);
      box.appendChild(item);
    });
    card.appendChild(box);
  }

  if (payload.tool_calls && payload.tool_calls.length) {
    const details = document.createElement("details");
    details.className = "tools";
    const summary = document.createElement("summary");
    summary.textContent = `本次调用了 ${payload.tool_calls.length} 个工具（点击查看输入与结果）`;
    details.appendChild(summary);
    const ul = document.createElement("ul");
    payload.tool_calls.forEach((call) => {
      const li = document.createElement("li");
      li.textContent = `${call.name} · ${call.ok ? "成功" : "失败"} · ${call.summary}`;
      ul.appendChild(li);
    });
    details.appendChild(ul);
    card.appendChild(details);
  }

  if (answer.need_followup) {
    const follow = document.createElement("p");
    follow.className = "muted small";
    follow.textContent = "需要你确认：" + answer.need_followup;
    card.appendChild(follow);
  }

  if (payload.run_id) {
    const rid = document.createElement("p");
    rid.className = "muted small";
    rid.textContent = "run_id: " + payload.run_id;
    card.appendChild(rid);
  }

  const wrapper = document.createElement("div");
  wrapper.className = "msg";
  wrapper.innerHTML = '<div class="avatar"></div>';
  wrapper.appendChild(card);
  el.messages.appendChild(wrapper);
  scrollToBottom();
}

function renderError(payload) {
  const card = document.createElement("div");
  card.className = "error-card";
  const title = document.createElement("h4");
  title.textContent = "本次请求没有完成：" + (payload.error || "未知错误");
  card.appendChild(title);

  // 错误分类由后端下发（AUTH/TIMEOUT/RATE_LIMIT/SCHEMA/ROUNDS…），
  // 便于一眼看出是"Key 过期"还是"模型抽风"，排查时不用逐字读错误信息。
  if (payload.error_code) {
    const code = document.createElement("span");
    code.className = "error-code";
    code.textContent = payload.error_code;
    card.appendChild(code);
  }

  const tip = document.createElement("p");
  tip.className = "small";
  tip.style.margin = "0";
  tip.textContent = "会话状态已保留，你可以修改成绩或换个问法后重新发送。";
  card.appendChild(tip);

  if (payload.error_detail) {
    const detail = document.createElement("div");
    detail.className = "error-detail";
    detail.textContent = payload.error_detail;
    card.appendChild(detail);
  }

  if (payload.tool_calls && payload.tool_calls.length) {
    const detail = document.createElement("div");
    detail.className = "error-detail";
    detail.textContent = "已执行的工具：" + payload.tool_calls.map((c) => `${c.name}(${c.ok ? "成功" : "失败"})`).join("、");
    card.appendChild(detail);
  }

  const wrapper = document.createElement("div");
  wrapper.className = "msg";
  wrapper.innerHTML = '<div class="avatar"></div>';
  wrapper.appendChild(card);
  el.messages.appendChild(wrapper);
  scrollToBottom();
}

function formatNum(value) {
  const num = Number(value);
  return Number.isFinite(num) ? (Math.round(num * 100) / 100).toString() : String(value);
}

/* ---------------- 请求 ---------------- */

/* 后端 SSE 推来的阶段 → 给人看的一句话 */
const PHASE_TEXT = {
  thinking: (event) => `模型正在思考（第 ${event.round} 轮）…`,
  tool: (event) => {
    if (event.by === "program") return "正在补充检索本地资料…";
    if (event.name === "analyze_score_trend") return "正在拟合成绩趋势…";
    if (event.name === "search_study_knowledge") return "正在检索本地资料…";
    return "正在调用工具…";
  },
  tool_done: (event) =>
    event.name === "search_study_knowledge" ? "已拿到资料，正在组织回答…" : "已算出结果，正在组织回答…",
  repair: () => "输出格式不合格，正在自动修正…",
};

/* 读 SSE：按空行切事件，取 data: 后的 JSON；任何解析失败都跳过，不影响主流程 */
async function consumeStream(response, loading) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  let failure = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const line = block.split("\n").find((item) => item.startsWith("data:"));
      if (!line) continue;
      let event;
      try {
        event = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }
      if (event.type === "progress") {
        const render = PHASE_TEXT[event.phase];
        if (render) loading.setPhase(render(event));
      } else if (event.type === "result") {
        result = event.payload;
      } else if (event.type === "error") {
        failure = event.message;
      }
    }
  }

  if (result) return { ok: true, payload: result };
  throw new Error(failure || "连接中断，没有收到完整结果");
}

/* 老接口兜底：流式不可用（代理缓冲、浏览器不支持 ReadableStream）时仍能正常对话 */
async function postChatFallback(body, headers) {
  const response = await fetch("/api/chat", { method: "POST", headers, body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    return { ok: false, payload, httpError: payload.detail || `HTTP ${response.status}` };
  }
  return { ok: true, payload };
}

async function requestChat(message, loading) {
  const body = JSON.stringify({ session_id: state.sessionId, message });
  const headers = { "Content-Type": "application/json" };

  let response;
  try {
    response = await fetch("/api/chat/stream", { method: "POST", headers, body });
  } catch {
    return postChatFallback(body, headers);
  }

  const contentType = response.headers.get("content-type") || "";
  if (!response.ok || !response.body || !contentType.includes("text/event-stream")) {
    return postChatFallback(body, headers);
  }
  return consumeStream(response, loading);
}

async function sendMessage(text) {
  if (state.sending) return;
  const message = text.trim();
  if (!message) return;

  state.sending = true;
  el.send.disabled = true;
  appendUserMessage(message);
  el.input.value = "";
  autoGrowInput();
  const loading = appendLoading();

  try {
    const outcome = await requestChat(message, loading);
    loading.stop();
    loading.wrapper.remove();

    if (!outcome.ok) {
      renderError({ error: outcome.httpError || "请求失败", error_detail: "请求被后端拒绝" });
      return;
    }

    const payload = outcome.payload;
    state.sessionId = payload.session_id;
    localStorage.setItem("ssa_session", state.sessionId);
    syncSessionUI();

    if (payload.error) {
      renderError(payload);
    } else {
      renderAnswer(payload);
      refreshHistoryView();
    }
  } catch (error) {
    loading.stop();
    loading.wrapper.remove();
    renderError({ error: "网络请求失败", error_detail: String(error) });
  } finally {
    state.sending = false;
    el.send.disabled = false;
    el.input.focus();
  }
}

/* ---------------- 历史记录 ---------------- */

async function loadHistory() {
  try {
    const response = await fetch("/api/history?limit=20&group=1");
    const data = await response.json();
    const sessions = data.sessions || [];
    const turns = sessions.reduce((sum, s) => sum + (s.turn_count || 0), 0);
    el.sessionCount.textContent = sessions.length
      ? `${sessions.length} 次会话 · ${turns} 轮`
      : "0 条分析";

    renderInsights(sessions);

    if (!sessions.length) {
      el.history.innerHTML = '<p class="muted small" style="padding:6px">暂无记录</p>';
      return;
    }
    el.history.innerHTML = "";
    sessions.forEach((session, index) => {
      el.history.appendChild(buildSessionNode(session, index === 0));
    });
  } catch (error) {
    el.history.innerHTML = '<p class="muted small" style="padding:6px">加载失败</p>';
  }
}

/* ---------------- 记录搜索：关键词 / 会话 ID / 时间 ---------------- */

const SEARCH_HINT_IDLE = "搜问题关键词；输入 8 位以上会话 ID 或 2026-10-30 这类日期会自动切换";
// 会话 ID 是 32 位十六进制（也可能带连字符）；日期形如 2026-10-30 / 2026-10 / 10-30
const SESSION_ID_RE = /^(?:[0-9a-f]{6,64}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$/i;
const DATE_RE = /^(?:\d{4}(?:[-/.]\d{1,2}){0,2}|\d{1,2}[-/.]\d{1,2})$/;

/* 判断这次输入想按什么搜：先认日期，再认会话 ID，其余都当关键词 */
function detectSearchMode(raw) {
  const text = String(raw || "").trim();
  if (!text) return { mode: "none", value: "" };
  if (DATE_RE.test(text)) return { mode: "date", value: text };
  if (SESSION_ID_RE.test(text)) return { mode: "session", value: text };
  return { mode: "keyword", value: text };
}

function setSearchHint(text, tone = "") {
  el.searchHint.textContent = text;
  el.searchHint.dataset.tone = tone;
  el.searchHint.classList.toggle("empty", tone === "warn");
}

/* 把命中的关键词标出来（只做高亮，textContent 拼接避免注入） */
function highlightText(text, keyword) {
  const frag = document.createDocumentFragment();
  const source = String(text || "");
  const key = String(keyword || "").trim();
  if (!key) {
    frag.appendChild(document.createTextNode(source));
    return frag;
  }
  const lower = source.toLowerCase();
  const target = key.toLowerCase();
  let from = 0;
  let index = lower.indexOf(target);
  while (index !== -1) {
    if (index > from) frag.appendChild(document.createTextNode(source.slice(from, index)));
    const mark = document.createElement("b");
    mark.className = "si-hit";
    mark.textContent = source.slice(index, index + target.length);
    frag.appendChild(mark);
    from = index + target.length;
    index = lower.indexOf(target, from);
  }
  frag.appendChild(document.createTextNode(source.slice(from)));
  return frag;
}

function buildSearchItem(record, keyword) {
  const item = document.createElement("div");
  item.className = "search-item";
  item.title = "点击查看完整问答";

  const top = document.createElement("div");
  top.className = "si-top";
  const turn = document.createElement("span");
  turn.className = "si-turn";
  turn.textContent = `第${record.turn_index || 1}轮`;
  const q = document.createElement("span");
  q.className = "si-q";
  q.appendChild(highlightText(record.question, keyword));
  top.append(turn, q);

  const meta = document.createElement("div");
  meta.className = "si-meta";
  meta.append(
    document.createTextNode(`会话 ${String(record.session_id || "").slice(0, 8)} · ${shortTime(record.created_at)} · `)
  );
  if (record.kind === "retrieval") {
    const tag = document.createElement("em");
    tag.textContent = retrievalLabel(record);
    meta.appendChild(tag);
  } else {
    meta.append(document.createTextNode(`斜率 ${formatNum(record.slope)}`));
    if (record.prediction !== null && record.prediction !== undefined) {
      meta.append(document.createTextNode(` · 预测 ${formatNum(record.prediction)} 分`));
    }
  }

  item.append(top, meta);
  item.addEventListener("click", () => showRecord(record));
  return item;
}

function renderSearchResults(data, mode, value) {
  const items = data.items || [];
  el.history.innerHTML = "";

  const scope =
    mode === "keyword" ? `关键词「${value}」`
      : mode === "session" ? `会话 ID「${value}」`
        : `时间「${value}」`;
  const capped = items.length >= 50 ? "（已到 50 条上限，可再加条件缩小范围）" : "";
  setSearchHint(`${scope} · 命中 ${items.length} 条记录${capped}`, items.length ? "active" : "warn");

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "search-empty";
    const title = document.createElement("b");
    title.textContent = "没有找到匹配的记录";
    empty.append(title, document.createElement("br"));
    empty.append(
      document.createTextNode(
        mode === "session" ? "会话 ID 是 32 位十六进制，输入前 8 位即可" : "换个关键词试试，或检查日期是否写对"
      )
    );
    el.history.appendChild(empty);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "search-results";
  items.forEach((record) => {
    wrap.appendChild(buildSearchItem(record, mode === "keyword" ? value : ""));
  });
  el.history.appendChild(wrap);
}

async function runSearch(raw) {
  const { mode, value } = detectSearchMode(raw);
  if (mode === "none") {
    clearSearch();
    return;
  }

  state.search.active = true;
  state.search.raw = raw;
  state.search.mode = mode;
  el.searchClear.hidden = false;
  el.historyTitle.textContent = "搜索结果";

  const params = new URLSearchParams();
  if (mode === "keyword") params.set("q", value);
  if (mode === "session") params.set("session_id", value);
  if (mode === "date") params.set("date", value);
  params.set("limit", "50");

  try {
    const response = await fetch(`/api/search?${params.toString()}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    renderSearchResults(data, mode, value);
  } catch (error) {
    el.history.innerHTML = "";
    const box = document.createElement("div");
    box.className = "search-empty";
    box.textContent = `搜索失败：${error.message || "请稍后重试"}`;
    el.history.appendChild(box);
    setSearchHint(`搜索失败：${error.message || "请稍后重试"}`, "warn");
  }
}

function clearSearch() {
  clearTimeout(state.search.timer);
  state.search.active = false;
  state.search.raw = "";
  el.searchInput.value = "";
  el.searchClear.hidden = true;
  el.historyTitle.textContent = "分析记录";
  setSearchHint(SEARCH_HINT_IDLE);
  loadHistory();
}

/* 刷新当前视图：正在搜索就重跑搜索，否则拉会话列表 */
async function refreshHistoryView() {
  if (state.search.active && el.searchInput.value.trim()) {
    await runSearch(el.searchInput.value);
    return;
  }
  await loadHistory();
}

/* ---------------- 数据看板 ---------------- */

function shortTime(iso) {
  if (!iso) return "—";
  const s = String(iso).replace("T", " ");
  return s.slice(5, 16);
}

function renderInsights(sessions) {
  const records = sessions.flatMap((s) => s.records || []);
  const trends = records
    .filter((r) => r.kind !== "retrieval" && r.kind !== "error")
    .slice()
    .sort((a, b) => String(a.created_at || "").localeCompare(String(b.created_at || "")));
  const retrievals = records.filter((r) => r.kind === "retrieval");

  el.kpiSessions.textContent = String(sessions.length);
  el.kpiTurns.textContent = String(sessions.reduce((sum, s) => sum + (s.turn_count || 0), 0));

  const slopes = trends.map((r) => Number(r.slope)).filter((v) => Number.isFinite(v));
  el.kpiSlope.classList.remove("up", "down");
  if (slopes.length) {
    const avg = slopes.reduce((a, b) => a + b, 0) / slopes.length;
    el.kpiSlope.textContent = (avg > 0 ? "+" : "") + formatNum(avg);
    el.kpiSlope.classList.add(avg >= 0 ? "up" : "down");
    el.kpiSlopeFoot.textContent = `${slopes.length} 条趋势记录`;
  } else {
    el.kpiSlope.textContent = "—";
    el.kpiSlopeFoot.textContent = "暂无趋势记录";
  }

  const latest = trends.filter((r) => r.prediction !== null && r.prediction !== undefined).pop();
  el.kpiPrediction.classList.remove("up", "down");
  if (latest) {
    el.kpiPrediction.textContent = formatNum(latest.prediction);
    el.kpiPredictionFoot.textContent = `第 ${latest.predict_x} 次 · ${shortTime(latest.created_at)}`;
  } else {
    el.kpiPrediction.textContent = "—";
    el.kpiPredictionFoot.textContent = "暂无预测";
  }

  const total = records.length;
  const trendCount = trends.length;
  const retrievalCount = retrievals.length;
  el.distTotal.textContent = `${total} 条`;
  el.valTrend.textContent = String(trendCount);
  el.valRetrieval.textContent = String(retrievalCount);
  const pct = (n) => `${total ? Math.round((n / total) * 100) : 0}%`;
  el.barTrend.style.width = pct(trendCount);
  el.barRetrieval.style.width = pct(retrievalCount);

  drawSparkline(slopes.slice(-10));

  const newest = sessions.reduce((acc, s) => {
    const ts = String(s.updated_at || "");
    return ts > acc ? ts : acc;
  }, "");
  el.insightUpdated.textContent = newest ? `更新 ${shortTime(newest)}` : "—";
}

function drawSparkline(values) {
  el.sparkline.innerHTML = "";
  el.sparkCaption.textContent = `最近 ${values.length} 条`;

  if (values.length < 2) {
    el.sparkEmpty.hidden = false;
    el.sparkEmpty.textContent = values.length
      ? "至少两条趋势记录才能画出走势"
      : "还没有趋势记录，先问一句试试";
    return;
  }
  el.sparkEmpty.hidden = true;

  const W = 268;
  const H = 92;
  const pad = 12;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = (W - pad * 2) / (values.length - 1);

  const points = values.map((v, i) => [
    pad + i * step,
    H - pad - ((v - min) / span) * (H - pad * 2),
  ]);

  let line = `M ${points[0][0].toFixed(2)} ${points[0][1].toFixed(2)}`;
  for (let i = 1; i < points.length; i += 1) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    const mx = (x0 + x1) / 2;
    line += ` C ${mx.toFixed(2)} ${y0.toFixed(2)}, ${mx.toFixed(2)} ${y1.toFixed(2)}, ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  }

  const last = points[points.length - 1];
  const first = points[0];
  const area = `${line} L ${last[0].toFixed(2)} ${H - pad} L ${first[0].toFixed(2)} ${H - pad} Z`;
  const dots = points
    .map(([x, y], i) =>
      i === points.length - 1
        ? `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="6" fill="rgba(79,124,255,.16)"/><circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="3" fill="#3f6df0"/>`
        : `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.8" fill="#fff" stroke="#8fa8e8" stroke-width="1.2"/>`
    )
    .join("");

  el.sparkline.innerHTML = `
    <defs>
      <linearGradient id="sparkStroke" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#4f7cff"/>
        <stop offset="60%" stop-color="#7b5cff"/>
        <stop offset="100%" stop-color="#22c1e0"/>
      </linearGradient>
      <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="rgba(79,124,255,.26)"/>
        <stop offset="100%" stop-color="rgba(79,124,255,0)"/>
      </linearGradient>
    </defs>
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="rgba(15,27,51,.07)" stroke-dasharray="3 4"/>
    <path d="${area}" fill="url(#sparkFill)"/>
    <path d="${line}" fill="none" stroke="url(#sparkStroke)" stroke-width="2" stroke-linecap="round"/>
    ${dots}
  `;
}

/* 删除单条分析记录：先确认，再调后端，最后刷新列表 */
async function deleteRecord(record) {
  const label = record.kind === "retrieval" ? "这条检索记录" : "这条分析记录";
  const ok = confirm(
    `确定删除${label}吗？\n\n「${record.question}」\n\n删除后不可恢复（只删这条历史记录，当前对话仍可继续追问）。`
  );
  if (!ok) return;

  try {
    const response = await fetch(`/api/history/${record.id}`, { method: "DELETE" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

    const dialog = document.querySelector(".record-dialog");
    if (dialog) dialog.close();

    await refreshHistoryView();
    const left = data.remaining || 0;
    setStatus(
      left ? `已删除 1 条记录，该会话还剩 ${left} 轮` : "已删除 1 条记录，这条会话已没有其他轮次",
      "ok"
    );
  } catch (error) {
    setStatus(`删除失败：${error.message || "请稍后重试"}`, "warn");
  }
}

/* 删除整个会话：确认后清空该会话的全部轮次；如果删的是当前会话，聊天区一并回到初始状态 */
async function deleteSession(session) {
  const turns = session.turn_count || (session.records || []).length || 1;
  const ok = confirm(
    `删除整个会话？\n\n「${session.title}」\n\n将删除该会话的全部 ${turns} 轮记录，删除后不可恢复。`
  );
  if (!ok) return;

  try {
    const response = await fetch(`/api/session/${session.session_id}`, { method: "DELETE" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

    const isCurrent = state.sessionId === session.session_id;
    await refreshHistoryView();

    if (isCurrent) {
      state.sessionId = null;
      localStorage.removeItem("ssa_session");
      el.messages.innerHTML = "";
      el.messages.appendChild(el.emptyState);
      syncSessionUI();
      setStatus(`已删除整个会话（${data.deleted || turns} 轮），已回到新会话`, "ok");
    } else {
      setStatus(`已删除整个会话（${data.deleted || turns} 轮记录）`, "ok");
    }
  } catch (error) {
    setStatus(`删除失败：${error.message || "请稍后重试"}`, "warn");
  }
}

/* 检索类记录的展示标签 */
function retrievalLabel(record) {
  if (!record.sources || !record.sources.length) return "资料不足";
  if (record.mode === "keyword") return "关键词检索";
  if (record.mode === "vector") return "向量检索";
  return "混合检索";
}

/* 一条会话 = 一张卡片：标题为第一轮提问，展开后按轮次列出 */
function buildSessionNode(session, expanded) {
  const card = document.createElement("div");
  card.className = "history-item session" + (expanded ? " open" : "");

  const head = document.createElement("div");
  head.className = "session-head";

  const q = document.createElement("div");
  q.className = "q";
  q.textContent = session.title;
  q.title = session.title;

  const meta = document.createElement("div");
  meta.className = "meta-line";
  const badge = session.turn_count > 1 ? `${session.turn_count} 轮追问` : "1 轮";
  const latest = session.records[session.records.length - 1] || {};
  const tail = session.kind === "retrieval"
    ? retrievalLabel(latest)
    : session.kind === "error"
      ? "调用失败"
      : `斜率 ${formatNum(session.slope)}`;
  meta.textContent = `${shortTime(session.updated_at)} · ${badge} · ${tail}`;

  // 操作行：继续追问 / 删除两个按钮同一行，固定排在日期下方
  const actions = document.createElement("div");
  actions.className = "session-actions";

  // 点开历史会话继续多轮问答（不重新输数据，追问直接复用该会话的上下文，
  // 聊天区会按时间顺序回放这个会话的全部轮次）
  const resume = document.createElement("button");
  resume.type = "button";
  resume.className = "session-resume";
  resume.textContent = "继续追问";
  resume.title = "重新打开这个会话，继续追问";
  resume.addEventListener("click", (event) => {
    event.stopPropagation();
    resumeSession(session.session_id);
  });

  // 删除整个会话：连同该会话的全部轮次一起删掉
  const delSession = document.createElement("button");
  delSession.type = "button";
  delSession.className = "session-del";
  delSession.title = "删除整个会话";
  delSession.setAttribute("aria-label", "删除整个会话");
  delSession.innerHTML = '<svg><use href="#i-trash"/></svg>';
  delSession.addEventListener("click", (event) => {
    event.stopPropagation();
    deleteSession(session);
  });

  actions.append(resume, delSession);
  head.append(q, meta, actions);

  const turnList = document.createElement("ul");
  turnList.className = "turn-list";
  (session.records || []).forEach((record, i) => {
    const li = document.createElement("li");
    li.className = "turn-item";
    const label = document.createElement("span");
    label.className = "turn-label";
    label.textContent = `第${i + 1}轮`;
    const text = document.createElement("span");
    text.className = "turn-text";
    text.textContent = record.question;
    text.title = record.question;
    li.append(label, text);
    if (record.kind === "retrieval") {
      const tag = document.createElement("span");
      tag.className = "turn-tag" + (record.sources && record.sources.length ? "" : " empty");
      tag.textContent = retrievalLabel(record);
      li.appendChild(tag);
    } else if (record.kind === "error") {
      const tag = document.createElement("span");
      tag.className = "turn-tag error";
      tag.textContent = "调用失败";
      li.appendChild(tag);
    }
    li.addEventListener("click", (event) => {
      event.stopPropagation();
      showRecord(record);
    });

    // 逐条删除：删掉某一次提问的记录，不影响同一会话里的其他轮次
    const del = document.createElement("button");
    del.type = "button";
    del.className = "turn-del";
    del.textContent = "×";
    del.title = "删除这条记录";
    del.setAttribute("aria-label", `删除第 ${i + 1} 轮记录`);
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteRecord(record);
    });
    li.appendChild(del);

    turnList.appendChild(li);
  });

  card.append(head, turnList);
  head.addEventListener("click", () => card.classList.toggle("open"));
  return card;
}

function formatTime(iso) {
  if (!iso) return "—";
  // 数据库存的是带时区的 ISO 时间（如 2026-11-02T17:19:38+08:00），
  // 展示时去掉 T 和时区后缀，只留「年月日 时分秒」；存储层仍保留完整时区信息
  return String(iso).replace("T", " ").replace(/([+-]\d{2}:?\d{2}|Z)$/, "").trim();
}

/* 早期记录没有保存回答原文时的兜底文案：只用已存的数值拼一句，不编造内容 */
function fallbackAnswer(record) {
  if (record.kind === "retrieval") {
    return (record.sources && record.sources.length)
      ? "（历史记录）检索了本地资料并命中相关片段。"
      : "（历史记录）知识库未覆盖该问题，属于资料不足。";
  }
  const parts = [];
  if (record.slope !== null && record.slope !== undefined) parts.push(`斜率 ${formatNum(record.slope)} 分/次`);
  if (record.intercept !== null && record.intercept !== undefined) parts.push(`截距 ${formatNum(record.intercept)}`);
  if (record.r2 !== null && record.r2 !== undefined) parts.push(`R² ${formatNum(record.r2)}`);
  if (record.prediction !== null && record.prediction !== undefined) {
    parts.push(`预测第 ${record.predict_x} 次为 ${formatNum(record.prediction)} 分`);
  }
  return `（早期记录未保存回答原文，仅保留计算结果：${parts.join("，") || "无拟合结果"}）`;
}

/* 点开历史会话继续多轮问答：把该会话的历史对话渲染回聊天区，并切换为当前会话 */
async function resumeSession(sessionId) {
  try {
    const response = await fetch(`/api/session/${sessionId}`);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      setStatus(data.detail || "恢复会话失败：记录可能已被清空", "warn");
      return;
    }
    const data = await response.json();
    state.sessionId = data.session_id;
    localStorage.setItem("ssa_session", data.session_id);
    syncSessionUI();

    el.messages.innerHTML = "";
    const note = document.createElement("div");
    note.className = "resume-note";
    note.textContent = `已重新打开这个会话（${data.turn_count} 轮记录${data.has_dataset ? "，成绩数据已保留，可直接问“第五次呢”" : ""}）`;
    el.messages.appendChild(note);

    (data.turns || []).forEach((turn) => renderStoredTurn(turn));
    scrollToBottom();
    setStatus(`已恢复会话 ${data.session_id.slice(0, 8)}，可以继续追问`, "ok");
    el.input.focus();
    refreshHistoryView();
    closeDrawer();
  } catch (error) {
    setStatus("恢复会话失败，请稍后重试", "warn");
  }
}

/* 把一条历史记录渲染成"提问 + 回答卡"，结构与实时回答保持一致 */
function renderStoredTurn(turn) {
  const userWrapper = document.createElement("div");
  userWrapper.className = "msg user";
  userWrapper.innerHTML = '<div class="avatar"></div><div class="bubble"></div>';
  userWrapper.querySelector(".bubble").textContent = turn.question;
  el.messages.appendChild(userWrapper);

  const card = document.createElement("div");
  card.className = "answer-card";

  const text = document.createElement("p");
  text.className = "answer-text";
  text.textContent = turn.answer || fallbackAnswer(turn);
  card.appendChild(text);

  if (turn.kind === "retrieval") {
    if (turn.sources && turn.sources.length) {
      const title = document.createElement("div");
      title.className = "section-title";
      title.textContent = `资料出处（${retrievalLabel(turn)}）`;
      card.appendChild(title);
      renderSourceList(card, turn.sources);
    }
  } else {
    const stats = document.createElement("div");
    stats.className = "stats";
    if (turn.slope !== null && turn.slope !== undefined) stats.appendChild(statBlock("斜率(分/次)", formatNum(turn.slope)));
    if (turn.intercept !== null && turn.intercept !== undefined) stats.appendChild(statBlock("截距", formatNum(turn.intercept)));
    if (turn.prediction !== null && turn.prediction !== undefined) stats.appendChild(statBlock("预测值", formatNum(turn.prediction)));
    if (stats.children.length) card.appendChild(stats);

    if (turn.chart_url) {
      const img = document.createElement("img");
      img.className = "chart";
      img.src = turn.chart_url;
      img.alt = "历史成绩趋势拟合图";
      card.appendChild(img);
    }
  }

  const time = document.createElement("p");
  time.className = "muted small";
  time.textContent = "记录时间：" + formatTime(turn.created_at);
  card.appendChild(time);

  const wrapper = document.createElement("div");
  wrapper.className = "msg";
  wrapper.innerHTML = '<div class="avatar"></div>';
  wrapper.appendChild(card);
  el.messages.appendChild(wrapper);
  scrollToBottom();
}

/* 需要用户补充/修正时的可操作卡片：预填原数据、标红错误项、给一键选项 */
function buildFixCard(fix) {
  const box = document.createElement("div");
  box.className = "fix-card";

  const title = document.createElement("div");
  title.className = "fix-title";
  title.textContent = fix.title || "需要你确认";
  box.appendChild(title);

  if (fix.hint) {
    const hint = document.createElement("p");
    hint.className = "fix-hint";
    hint.textContent = fix.hint;
    box.appendChild(hint);
  }

  // ---- 选择预测位置：直接点按钮 ----
  if (fix.kind === "choose_predict") {
    const row = document.createElement("div");
    row.className = "fix-options";
    const scores = (fix.scores || []).map((v) => String(v).trim()).filter(Boolean);
    (fix.options || []).forEach((option) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "fix-option";
      btn.textContent = `第 ${option} 次`;
      btn.addEventListener("click", () => {
        const prefix = scores.length ? `我的成绩依次是 ${scores.join("、")}，` : "";
        sendMessage(`${prefix}帮我预测第 ${option} 次考试的成绩`);
        row.querySelectorAll("button").forEach((b) => (b.disabled = true));
      });
      row.appendChild(btn);
    });
    box.appendChild(row);
    return box;
  }

  // ---- 编辑 / 追加成绩 ----
  const values = (fix.scores || []).map((v) => String(v ?? ""));
  const invalid = new Set(fix.invalid || []);
  const grid = document.createElement("div");
  grid.className = "fix-grid";
  const inputs = [];

  const addRow = (value, label, isInvalid) => {
    const cell = document.createElement("label");
    cell.className = "fix-cell" + (isInvalid ? " invalid" : "");
    const span = document.createElement("span");
    span.textContent = label;
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.value = value;
    input.placeholder = "0—150";
    cell.append(span, input);
    grid.appendChild(cell);
    inputs.push(input);
    return cell;
  };

  values.forEach((value, index) => addRow(value, `第 ${index + 1} 次`, invalid.has(index + 1)));

  // 数据不足时补两个空位，让"补充成绩"变成填空而不是重新打字
  const blanks = fix.kind === "add_scores" ? Math.max(1, 2 - values.length) : 0;
  for (let i = 0; i < blanks; i += 1) {
    addRow("", `第 ${values.length + i + 1} 次`, false);
  }
  box.appendChild(grid);

  if (fix.regenerate_index && values.length) {
    const note = document.createElement("p");
    note.className = "fix-note";
    note.textContent = "考试序号会按 1、2、3… 自动重排，你只需要核对成绩。";
    box.appendChild(note);
  }

  const actions = document.createElement("div");
  actions.className = "fix-actions";

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "fix-link";
  addBtn.textContent = "+ 再加一次成绩";
  addBtn.addEventListener("click", () => addRow("", `第 ${inputs.length + 1} 次`, false));
  actions.appendChild(addBtn);

  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "fix-submit";
  submit.textContent = "重新分析";
  actions.appendChild(submit);
  box.appendChild(actions);

  const tip = document.createElement("p");
  tip.className = "fix-tip";
  box.appendChild(tip);

  submit.addEventListener("click", () => {
    const scores = [];
    for (const input of inputs) {
      const raw = input.value.trim();
      if (!raw) continue;
      if (!/^\d+(\.\d+)?$/.test(raw) || Number(raw) > 150) {
        input.parentElement.classList.add("invalid");
        tip.textContent = `「${raw}」不是有效成绩：请填 0—150 之间的数字`;
        return;
      }
      input.parentElement.classList.remove("invalid");
      scores.push(Number(raw));
    }
    if (scores.length < 2) {
      tip.textContent = "至少需要 2 次成绩才能拟合趋势，请再补一个";
      return;
    }
    const tail = fix.predict_x ? `，并预测第 ${fix.predict_x} 次的成绩` : "";
    submit.disabled = true;
    sendMessage(`我的成绩依次是 ${scores.join("、")}${tail}，帮我分析变化趋势`);
  });

  return box;
}

/* 记录详情弹窗：结构与整页视觉统一，不再用内联样式 */
function dialogRow(label, value) {
  const p = document.createElement("p");
  p.className = "dialog-row";
  const b = document.createElement("b");
  b.textContent = `${label}：`;
  p.append(b, document.createTextNode(value));
  return p;
}

function showRecord(item) {
  const existing = document.querySelector(".record-dialog");
  if (existing) existing.remove();

  const isError = item.kind === "error";
  const isRetrieval = item.kind === "retrieval";
  const dialog = document.createElement("dialog");
  dialog.className = "record-dialog";

  const head = document.createElement("div");
  head.className = "dialog-head";
  const headText = document.createElement("div");
  const kicker = document.createElement("div");
  kicker.className = "dialog-kicker";
  kicker.textContent = isError ? "Failure" : isRetrieval ? "Retrieval" : "Trend Analysis";
  const title = document.createElement("h3");
  title.className = "dialog-title";
  title.textContent = (isError ? "失败记录 #" : isRetrieval ? "检索记录 #" : "分析记录 #") + item.id;
  headText.append(kicker, title);
  head.appendChild(headText);

  if (isError) {
    const tag = document.createElement("span");
    tag.className = "turn-tag error";
    tag.textContent = "调用失败";
    head.appendChild(tag);
  } else if (isRetrieval) {
    const tag = document.createElement("span");
    tag.className = "turn-tag" + (item.sources && item.sources.length ? "" : " empty");
    tag.textContent = retrievalLabel(item);
    head.appendChild(tag);
  }

  const body = document.createElement("div");
  body.className = "dialog-body";
  body.appendChild(dialogRow("问题", item.question));

  if (isError) {
    // 失败记录：按「标题：内容」拆行展示失败原因与排查信息，不显示任何数值结果
    (item.answer || "").split("\n").filter(Boolean).forEach((line) => {
      const sep = line.indexOf("：");
      const label = sep > 0 ? line.slice(0, sep) : "";
      const value = sep > 0 ? line.slice(sep + 1) : line;
      body.appendChild(dialogRow(label, value));
    });
    body.appendChild(dialogRow("时间", formatTime(item.created_at)));
  } else if (isRetrieval) {
    body.appendChild(dialogRow("检索方式", `${retrievalLabel(item)}（mode=${item.mode || "hybrid"}）`));
    body.appendChild(dialogRow("回答", item.answer || "—"));
    body.appendChild(dialogRow("时间", formatTime(item.created_at)));
    if (item.sources && item.sources.length) {
      const srcTitle = document.createElement("div");
      srcTitle.className = "section-title";
      srcTitle.textContent = "命中片段";
      body.appendChild(srcTitle);
      renderSourceList(body, item.sources);
    }
  } else {
    body.appendChild(dialogRow("回答", item.answer || fallbackAnswer(item)));
    body.appendChild(dialogRow("数据", `x=${JSON.stringify(item.x)}，y=${JSON.stringify(item.y)}`));
    body.appendChild(dialogRow("斜率 / 截距", `${formatNum(item.slope)} / ${formatNum(item.intercept)}`));
    body.appendChild(dialogRow("预测", `第 ${item.predict_x ?? "—"} 次 → ${item.prediction ?? "未预测"}`));
    if (item.pred_low !== null && item.pred_low !== undefined && item.pred_high !== null && item.pred_high !== undefined) {
      body.appendChild(
        dialogRow("95% 预测区间", `${formatNum(item.pred_low)} — ${formatNum(item.pred_high)} 分`),
      );
    }
    body.appendChild(dialogRow("R²", formatNum(item.r2)));
    body.appendChild(dialogRow("时间", formatTime(item.created_at)));

    if (item.chart_url) {
      const img = document.createElement("img");
      img.src = item.chart_url;
      img.alt = "趋势拟合图";
      img.className = "chart";
      img.style.marginTop = "10px";
      body.appendChild(img);
    }
  }

  const foot = document.createElement("div");
  foot.className = "dialog-foot";

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "record-delete";
  remove.textContent = "删除这条记录";
  remove.addEventListener("click", () => deleteRecord(item));

  const close = document.createElement("button");
  close.type = "button";
  close.className = "dialog-close";
  close.textContent = "关闭";
  close.addEventListener("click", () => dialog.close());

  foot.append(remove, close);
  dialog.append(head, body, foot);
  document.body.appendChild(dialog);
  dialog.showModal();
  dialog.addEventListener("close", () => dialog.remove());
}

function renderSourceList(container, sources) {
  const box = document.createElement("div");
  box.className = "sources";
  box.style.marginTop = "8px";
  (sources || []).forEach((src) => {
    const item = document.createElement("div");
    item.className = "source";
    const head = document.createElement("div");
    head.className = "src-head";
    head.textContent = `${src.title} · ${src.chunk_id}`;
    const snippet = document.createElement("div");
    snippet.textContent = src.snippet;
    item.append(head, snippet);
    box.appendChild(item);
  });
  container.appendChild(box);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

/* ---------------- 界面交互 ---------------- */

function updateScrollButton() {
  if (!el.scrollBottom) return;
  const distance = el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight;
  el.scrollBottom.hidden = distance < 180;
}

function autoGrowInput() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(Math.max(el.input.scrollHeight, 42), 144)}px`;
}

function openDrawer() {
  el.app.classList.add("drawer-open");
}

function closeDrawer() {
  el.app.classList.remove("drawer-open");
}

/* 滚动渐入：看板卡片进入视口时上浮淡入 */
function initReveal() {
  const items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    items.forEach((node) => node.classList.add("in"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  items.forEach((node) => observer.observe(node));
}

/* ---------------- 事件绑定 ---------------- */

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(el.input.value);
});

el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(el.input.value);
  }
});

el.input.addEventListener("input", autoGrowInput);

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const text = chip.dataset.text || chip.textContent;
    el.input.value = text;
    autoGrowInput();
    sendMessage(text);
  });
});

el.messages.addEventListener("scroll", updateScrollButton, { passive: true });

if (el.scrollBottom) {
  el.scrollBottom.addEventListener("click", scrollToBottom);
}

if (el.sidebarOpen) el.sidebarOpen.addEventListener("click", openDrawer);
if (el.sidebarClose) el.sidebarClose.addEventListener("click", closeDrawer);

/* 搜索框：输入即搜（防抖 280ms），回车立即搜，Esc 或 × 清除 */
el.searchInput.addEventListener("input", () => {
  const raw = el.searchInput.value;
  el.searchClear.hidden = !raw.trim();
  clearTimeout(state.search.timer);
  if (!raw.trim()) {
    clearSearch();
    return;
  }
  state.search.timer = setTimeout(() => runSearch(el.searchInput.value), 280);
});

el.searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    clearTimeout(state.search.timer);
    runSearch(el.searchInput.value);
  }
  if (event.key === "Escape") {
    event.preventDefault();
    clearSearch();
  }
});

el.searchClear.addEventListener("click", () => {
  clearSearch();
  el.searchInput.focus();
});

el.refresh.addEventListener("click", refreshHistoryView);

el.clearHistory.addEventListener("click", async () => {
  if (!confirm("清空所有分析记录？此操作不可恢复（当前会话仍可继续追问）。")) return;
  try {
    const response = await fetch("/api/history?source=web", { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) throw new Error("清空失败");
    loadHistory();
    setStatus(`已清空 ${data.deleted || 0} 条记录`, "ok");
  } catch (error) {
    setStatus("清空失败，请稍后重试", "warn");
  }
});

el.reset.addEventListener("click", async () => {
  if (state.sessionId) {
    await fetch(`/api/session/${state.sessionId}/reset`, { method: "POST" });
  }
  state.sessionId = null;
  localStorage.removeItem("ssa_session");
  el.messages.innerHTML = "";
  el.messages.appendChild(el.emptyState);
  syncSessionUI();
  setStatus("已开启新会话", "ok");
  loadHistory();
  closeDrawer();
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.mock_mode) {
      setStatus("桩模型模式（未接真实模型）", "warn");
    } else if (data.api_key_configured) {
      setStatus(`已连接 ${data.model}`, "ok");
    } else {
      setStatus("未配置 API Key", "err");
    }
  } catch (error) {
    setStatus("后端未启动", "err");
  }
}

syncSessionUI();
checkHealth();
loadHistory();
initReveal();
updateScrollButton();
el.input.focus();
