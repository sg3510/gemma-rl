"use strict";

const D = JSON.parse(document.getElementById("payload").textContent);
const TASKS = new Map(D.tasks.map((t) => [t.id, t]));
const BY_TASK = new Map();
for (const a of D.attempts) {
  if (!BY_TASK.has(a.task_id)) BY_TASK.set(a.task_id, []);
  BY_TASK.get(a.task_id).push(a);
}
for (const list of BY_TASK.values()) list.sort((x, y) => x.trial - y.trial);

const GROUP_KEY = D.run.primary_group || null;
const SYM = { pass: "✓", fail: "✗", error: "!", missing: "·" };
const LABEL = {
  pass: "solved",
  fail: "failed the task",
  error: "infrastructure error — not scored, needs a re-run",
  missing: "never run",
};

const state = { split: "all", group: "all", status: "all", q: "", sel: null, cmp: null };

// Loop flag mirrors taurl/loops.py's finding: failing lock-ups run >=5 identical
// calls back-to-back (no passing trace exceeds 4 consecutive).
const isLooped = (a) => ((a.quality || {}).max_consecutive_repeat || 0) >= 5;

const esc = (s) =>
  String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const pct = (v) => (v === null || v === undefined ? "n/a" : (v * 100).toFixed(1) + "%");
const num = (v, d = 1) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));

/* ---------------------------------------------------------------- header */

function renderTop() {
  const r = D.run, s = D.summary;
  const splitBadges = Object.entries(r.splits)
    .map(([k, n]) => `<span class="badge split-${esc(k)}">${esc(k)} ×${n}</span>`)
    .join("");
  const stack = r.stack || {};
  const temp = (stack.agent_args || {}).temperature;
  const notes = (r.notes || []).map((n) => `<p class="note">${esc(n)}</p>`).join("");
  const incomplete = s.num_tasks - s.full_depth_tasks;

  document.getElementById("top").innerHTML = `
    <h1>${esc(r.name)}</h1>
    <p class="sub">
      ${esc(r.family)} / ${esc(r.domain)} &middot; ${s.num_tasks} tasks &times;
      ${r.trials_planned} planned trials
      ${r.timestamp ? "&middot; " + esc(r.timestamp.slice(0, 16).replace("T", " ")) : ""}
      &middot; <a href="index.html">all runs</a>
    </p>
    <div class="badges">
      ${splitBadges}
      <span class="badge">agent ${esc(stack.agent_model || "?")}</span>
      ${temp !== undefined && temp !== null ? `<span class="badge">temp ${esc(temp)}</span>` : ""}
      <span class="badge">customer ${esc(stack.user_model || "?")}</span>
      ${stack.max_steps ? `<span class="badge">step cap ${esc(stack.max_steps)}</span>` : ""}
      ${s.errors ? `<span class="badge warn">${s.errors} infra errors</span>` : ""}
      ${s.missing ? `<span class="badge warn">${s.missing} never run</span>` : ""}
    </div>
    ${notes}
    <div class="stats">
      <div class="stat"><div class="n">${pct(s.pass1_macro)}</div>
        <div class="l">pass@1 macro</div></div>
      <div class="stat"><div class="n">${pct(s.pass1_micro)}</div>
        <div class="l">pass@1 micro</div></div>
      <div class="stat"><div class="n">${s.scored}<span class="l"> / ${s.cells}</span></div>
        <div class="l">rollouts scored</div></div>
      <div class="stat"><div class="n${incomplete ? " warn" : ""}">${s.full_depth_tasks} / ${s.num_tasks}</div>
        <div class="l">tasks at full depth</div></div>
      <div class="stat"><div class="n">${s.frontier_tasks}</div>
        <div class="l">frontier tasks (${pct(s.frontier_share)})</div></div>
    </div>
    <details><summary class="sub">pass@k, termination reasons${GROUP_KEY ? ", per-" + esc(GROUP_KEY) + " ladder" : ""}</summary>
      ${curveTable(s)}${terminationTable(s)}${groupTable(s)}${qualityTable(s)}
    </details>`;
}

function curveTable(s) {
  if (!s.curve.length) return "";
  const rows = s.curve.map((c) => `<tr>
      <td class="num">${c.k}</td>
      <td class="num">${pct(c.pass_at_k)}</td>
      <td class="num">${pct(c.pass_hat_k)}</td>
      <td class="num">${c.eligible_tasks} / ${s.num_tasks}</td>
    </tr>`).join("");
  return `<h3>pass@k</h3><table>
    <tr><th class="num">k</th><th class="num">pass@k</th>
        <th class="num">pass^k</th><th class="num">tasks with k scored rollouts</th></tr>
    ${rows}</table>
    <p class="legend">pass@k is computed only over tasks that actually have k scored
    rollouts, so read it together with the last column — pass@4 over a third of
    the tasks is not pass@4 over the run.</p>`;
}

function terminationTable(s) {
  const entries = Object.entries(s.termination || {});
  if (!entries.length) return "";
  const total = entries.reduce((acc, [, n]) => acc + n, 0);
  return `<h3>termination reasons</h3><table>${entries
    .map(([k, n]) => `<tr><td><code>${esc(k)}</code></td><td class="num">${n}</td>
      <td class="num">${pct(n / total)}</td></tr>`)
    .join("")}</table>`;
}

function groupTable(s) {
  if (!GROUP_KEY) return "";
  const rows = Object.entries(s.groups[GROUP_KEY])
    .map(([k, g]) => `<tr><td>${esc(k)}</td><td class="num">${g.tasks}</td>
      <td class="num">${pct(g.pass1_macro)}</td>
      <td class="num">${g.frontier_tasks} (${pct(g.frontier_share)})</td></tr>`)
    .join("");
  return `<h3>by ${esc(GROUP_KEY)}</h3><table>
    <tr><th>${esc(GROUP_KEY)}</th><th class="num">tasks</th>
        <th class="num">pass@1</th><th class="num">frontier</th></tr>${rows}</table>`;
}

function qualityTable(s) {
  const q = s.quality || {};
  if (!q.successes_with_traces) return "";
  return `<h3>trace quality, successes only</h3><table>
    <tr><td>successful traces</td><td class="num">${q.successes_with_traces}</td></tr>
    <tr><td>median messages</td><td class="num">${num(q.median_messages, 0)}</td></tr>
    <tr><td>median tool calls</td><td class="num">${num(q.median_tool_calls, 0)}</td></tr>
    <tr><td>traces with an exact duplicate call</td><td class="num">${q.traces_with_duplicates}</td></tr>
    <tr><td>median duplicate calls</td><td class="num">${num(q.median_duplicate_calls, 0)}</td></tr>
    </table>
    <p class="legend">Tier-1 signals from reports/tiered_reward.md: quality only ever
    breaks ties among successes, never overrides the success gate.</p>`;
}

/* ------------------------------------------------------------------ grid */

function visibleTasks() {
  const q = state.q.toLowerCase();
  return D.tasks.filter((t) => {
    if (state.split !== "all" && t.split !== state.split) return false;
    if (GROUP_KEY && state.group !== "all" && String(t.groups[GROUP_KEY]) !== state.group)
      return false;
    if (state.status === "frontier" && !t.stats.frontier) return false;
    if (state.status === "looped" &&
        !(BY_TASK.get(t.id) || []).some(isLooped)) return false;
    if (state.status === "dpo-pairable" && !t.stats.dpo_pairable) return false;
    if (state.status === "all-pass" && !(t.stats.n && t.stats.c === t.stats.n)) return false;
    if (state.status === "incomplete" && !t.stats.errors && !t.stats.missing) return false;
    if (state.status === "unsolved" && t.stats.c > 0) return false;
    if (state.status === "solved" && t.stats.c === 0) return false;
    if (q && !(t.id + " " + t.short + " " + t.goal).toLowerCase().includes(q)) return false;
    return true;
  });
}

function renderLeft() {
  const splits = [...new Set(D.tasks.map((t) => t.split).filter(Boolean))];
  const groups = GROUP_KEY
    ? [...new Set(D.tasks.map((t) => t.groups[GROUP_KEY]).filter((g) => g !== undefined))]
    : [];
  const tasks = visibleTasks();

  let rows = "", lastGroup = null;
  for (const t of tasks) {
    const g = GROUP_KEY ? String(t.groups[GROUP_KEY]) : null;
    if (g !== null && g !== lastGroup) {
      lastGroup = g;
      rows += `<tr class="group"><td colspan="4">${esc(g)}</td></tr>`;
    }
    const st = t.stats;
    rows += `<tr class="${t.stats.frontier ? "frontier" : ""}">
      <td class="task"><div class="short"><span class="ref" title="click to copy this task's reference">${esc(t.ref)}</span> ${esc(t.short)}</div>
        <div class="sub">${t.split ? esc(t.split) : ""}${
          t.groups.persona ? " &middot; persona " + esc(t.groups.persona) : ""
        }</div></td>
      <td class="cells">${(BY_TASK.get(t.id) || []).map(cellHTML).join("")}</td>
      <td class="num">${st.n ? st.c + "/" + st.n : "—"}</td>
      <td class="tags">${st.dpo_pairable ? `<span class="badge tag-dpo" title="two successes differ by &ge;3 duplicate calls — a Tier-1 preference pair">DPO</span>` : ""}${
        st.has_loop ? `<span class="badge tag-loop" title="a rollout repeated one call &ge;10 times">loop</span>` : ""}${
        st.duplicate_spread ? `<span class="badge" title="difference in duplicate calls between the best and worst success">&Delta;${st.duplicate_spread}</span>` : ""}</td>
    </tr>`;
  }

  document.getElementById("left").innerHTML = `
    <div class="controls">
      <label>split <select id="f-split">${opts(["all", ...splits], state.split)}</select></label>
      ${GROUP_KEY ? `<label>${esc(GROUP_KEY)} <select id="f-group">${opts(
        ["all", ...groups.map(String)], state.group)}</select></label>` : ""}
      <label>show <select id="f-status">${opts(
        ["all", "frontier", "dpo-pairable", "all-pass", "looped", "incomplete",
         "unsolved", "solved"], state.status)}</select></label>
      <input type="search" id="f-q" placeholder="search tasks" value="${esc(state.q)}">
    </div>
    <table class="grid">
      <tr><th>task (${tasks.length} of ${D.tasks.length})</th>
          <th>trials 0–${D.run.trials_planned - 1}</th><th class="num">c/n</th>
          <th>tags</th></tr>
      ${rows || `<tr><td colspan="4" class="empty">no tasks match</td></tr>`}
    </table>
    <p class="legend">
      <span class="cell pass">${SYM.pass}</span> solved
      <span class="cell fail">${SYM.fail}</span> failed
      <span class="cell error">${SYM.error}</span> infrastructure error, not scored
      <span class="cell missing">${SYM.missing}</span> never run
      <span class="cell fail loop">↻</span> tool-call loop (&ge;5 back-to-back identical calls) &mdash;
      purple bar marks a frontier task (pass@1 in [0.25, 0.75]).
    </p>`;

  bind("f-split", "split");
  bind("f-group", "group");
  bind("f-status", "status");
  const q = document.getElementById("f-q");
  q.oninput = () => { state.q = q.value; renderLeft(); q.focus(); };
  for (const el of document.querySelectorAll(".cell[data-task]")) {
    el.onclick = () => select(el.dataset.task, Number(el.dataset.trial));
  }
}

function opts(values, current) {
  return values
    .map((v) => `<option value="${esc(v)}"${v === current ? " selected" : ""}>${esc(v)}</option>`)
    .join("");
}

function bind(id, key) {
  const el = document.getElementById(id);
  if (el) el.onchange = () => { state[key] = el.value; renderLeft(); };
}

function cellHTML(a) {
  const sel = state.sel && state.sel.task === a.task_id && state.sel.trial === a.trial;
  const cmp = state.cmp && state.cmp.task === a.task_id && state.cmp.trial === a.trial;
  const loop = isLooped(a);
  const label = LABEL[a.status] + (loop
    ? ` — LOOPED (${a.quality.max_consecutive_repeat} identical calls back-to-back)` : "");
  return `<span class="cell ${a.status}${loop ? " loop" : ""}${sel || cmp ? " sel" : ""}"
    data-task="${esc(a.task_id)}" data-trial="${a.trial}"
    title="trial ${a.trial}: ${esc(label)}">${loop ? "↻" : SYM[a.status]}</span>`;
}

/* ---------------------------------------------------------------- detail */

function updateHash(taskId, trial) {
  const t = TASKS.get(taskId);
  if (t) history.replaceState(null, "", `#${t.ref}.${trial}`);
}

function fromHash() {
  const m = (location.hash || "").match(/^#(T-[0-9a-f]{4})\.(\d+)$/);
  if (!m) return null;
  const t = D.tasks.find((x) => x.ref === m[1]);
  return t ? { task: t.id, trial: Number(m[2]) } : null;
}

function select(taskId, trial) {
  if (state.sel && state.sel.task === taskId && state.sel.trial === trial) return;
  if (state.sel && state.sel.task !== taskId) state.cmp = null;
  state.sel = { task: taskId, trial };
  updateHash(taskId, trial);
  renderLeft();
  renderRight();
}

function attemptOf(taskId, trial) {
  return (BY_TASK.get(taskId) || []).find((a) => a.trial === trial);
}

function renderRight() {
  const pane = document.getElementById("right");
  if (!state.sel) {
    pane.innerHTML = `<p class="empty">Pick a cell to see the trace, and how the
      attempt scored against the task's ideal end state.</p>`;
    return;
  }
  const task = TASKS.get(state.sel.task);
  const a = attemptOf(state.sel.task, state.sel.trial);
  const b = state.cmp ? attemptOf(state.cmp.task, state.cmp.trial) : null;
  const others = (BY_TASK.get(task.id) || []).filter((x) => x.trial !== state.sel.trial);

  pane.innerHTML = `
    <h2>${esc(task.short)}</h2>
    <p class="sub">
      <button class="refbtn" id="copy-ref" title="copy a reference you can paste into chat">${
        esc(D.run.name)}/${esc(task.ref)}#${state.sel.trial}</button>
    </p>
    <p class="sub"><code>${esc(task.id)}</code></p>
    <div class="badges">
      ${(task.splits || []).map((s) => `<span class="badge split-${esc(s)}">${esc(s)}</span>`).join("")}
      ${Object.entries(task.groups).map(([k, v]) =>
        `<span class="badge">${esc(k)}: ${esc(v)}</span>`).join("")}
      ${task.reward_basis.length
        ? `<span class="badge">graded on ${task.reward_basis.map(esc).join(" × ")}</span>`
        : ""}
    </div>
    <details><summary class="sub">what the customer wants</summary>
      <div class="msg"><div class="body">${esc(task.goal || "(no scenario text)")}</div></div>
    </details>
    ${idealBlock(task)}
    <div class="controls">
      <label>compare with
        <select id="cmp">${opts(["none", ...others.map((o) => "trial " + o.trial)],
          state.cmp ? "trial " + state.cmp.trial : "none")}</select></label>
    </div>
    ${b ? `<div class="compare"><div>${attemptBlock(a, task)}</div>
              <div>${attemptBlock(b, task)}</div></div>`
        : attemptBlock(a, task)}`;

  const copy = document.getElementById("copy-ref");
  if (copy) copy.onclick = () => {
    navigator.clipboard.writeText(copy.textContent.trim() + "  " + location.href.split("#")[0]
      + "#" + task.ref + "." + state.sel.trial);
    copy.textContent = "copied ✓";
    setTimeout(() => renderRight(), 900);
  };

  const sel = document.getElementById("cmp");
  if (sel) sel.onchange = () => {
    state.cmp = sel.value === "none"
      ? null
      : { task: task.id, trial: Number(sel.value.split(" ")[1]) };
    renderLeft();
    renderRight();
  };
  for (const el of pane.querySelectorAll(".cell[data-task]")) {
    el.onclick = () => select(el.dataset.task, Number(el.dataset.trial));
  }
  pane.scrollTop = 0;
}

function idealBlock(task) {
  if (!task.ideal.length) return "";
  return `<details><summary class="sub">ideal end state — ${task.ideal.length}
    criteria from the task definition</summary>
    <ul class="checks">${task.ideal.map((c) => `<li>
      <span class="mark unk">○</span>${esc(c.label)}
      <span class="kind">${esc(c.kind)}</span>
      ${c.counts ? "" : `<span class="kind diag">diagnostic only</span>`}</li>`).join("")}</ul>
    <p class="legend">Only criteria in the reward basis gate the score; the rest
    still run, as diagnostics.</p></details>`;
}

function attemptBlock(a, task) {
  if (!a) return `<p class="empty">no such attempt</p>`;
  const q = a.quality;
  const head = `<h3>trial ${a.trial} — ${esc(LABEL[a.status])}</h3>
    <p class="sub">
      ${a.reward !== undefined ? "reward " + num(a.reward, 2) : ""}
      ${a.termination_reason ? " &middot; " + esc(a.termination_reason) : ""}
      ${a.duration ? " &middot; " + num(a.duration, 0) + "s" : ""}
      ${a.cost ? " &middot; $" + num(a.cost, 3) : ""}
      ${q ? ` &middot; ${q.messages} messages, ${q.tool_calls} tool calls` +
            (q.duplicate_calls ? `, <b>${q.duplicate_calls} duplicate</b>` : "") +
            (q.max_consecutive_repeat >= 5
              ? `, <b>looped ×${q.max_consecutive_repeat} consecutive</b>` : "") : ""}
    </p>
    ${a.note ? `<p class="note">${esc(a.note)}</p>` : ""}`;

  if (a.status === "missing")
    return head + `<p class="empty">This cell was never run. Re-running it is what
      turns pass@${task.stats.n + 1} from an estimate into a measurement.</p>`;

  return head + checksBlock(a, task) + traceBlock(a);
}

function checksBlock(a, task) {
  if (!a.checks || !a.checks.length) {
    // tau2 skips evaluation entirely when a simulation ends early, so reward
    // 0.0 here means "never scored", not "failed every check".
    return `<h3>verifier checks</h3>
      <p class="sub">This attempt was never scored — the simulation ended before
      the verifier ran, so its 0.0 is an absence of evidence, not a failed check.
      What it would have been graded on:</p>
      <ul class="checks">${(task.ideal || []).map((c) => `<li class="${
        c.counts ? "" : "diag"}"><span class="mark unk">○</span>${esc(c.label)}
        <span class="kind">${esc(c.kind)}</span></li>`).join("")
        || `<li class="sub">the task defines no criteria</li>`}</ul>`;
  }
  const mark = (m) => (m === true
    ? `<span class="mark ok">✓</span>`
    : m === false ? `<span class="mark no">✗</span>`
    : `<span class="mark unk">○</span>`);
  const items = a.checks.map((c) => `<li class="${c.counts ? "" : "diag"}">
      ${mark(c.met)}${esc(c.label)}
      <span class="kind">${esc(c.kind)}</span>
      ${c.counts ? "" : `<span class="kind diag">diagnostic only</span>`}
      ${c.detail && c.detail.justification
        ? `<div class="sub">${esc(c.detail.justification)}</div>` : ""}
    </li>`).join("");
  const breakdown = Object.entries(a.reward_breakdown || {})
    .map(([k, v]) => `${esc(k)} ${num(v, 2)}`).join(" &times; ");
  return `<h3>verifier checks</h3><ul class="checks">${items}</ul>
    ${breakdown ? `<p class="sub">reward = ${breakdown}</p>` : ""}`;
}

function traceBlock(a) {
  if (!a.messages || !a.messages.length)
    return `<p class="sub">no trace saved for this attempt</p>`;
  // Tools the agent tried to call that it was never given. In telecom these are
  // the CUSTOMER's device tools: the agent must ask the customer to run them,
  // but the tech-support manual names them in callable syntax, so the model
  // often calls them directly and gets "Tool not found".
  const HALLUCINATED = new Set();
  for (const m of a.messages) {
    const hit = /Tool '([^']+)' not found/.exec(m.content || "");
    if (hit) HALLUCINATED.add(hit[1]);
  }
  const seen = new Set();
  const parts = a.messages.map((m) => {
    const who = m.role === "tool"
      ? "tool result" + (m.requestor && m.requestor !== "assistant" ? " (customer)" : "")
      : m.role === "user" ? "customer" : m.role;
    let body = m.content ? `<div class="body">${esc(m.content)}</div>` : "";
    if (m.role === "tool") {
      const theirs = m.requestor && m.requestor !== "assistant";
      const bad = /Tool '[^']+' not found/.test(m.content || "");
      body = `<div class="call result${theirs ? " theirs" : ""}${bad ? " halluc" : ""}">`
        + `<pre>${esc(m.content)}</pre></div>`;
    }
    const calls = (m.tool_calls || []).map((c) => {
      const mine = !c.requestor || c.requestor === "assistant";
      const key = c.name + "::" + JSON.stringify(c.arguments || {});
      // Only the AGENT's calls can be duplicates worth flagging: the customer
      // re-checking their own device after a change is correct behaviour, and
      // the customer is a model we neither control nor train.
      const dup = mine && seen.has(key);
      if (mine) seen.add(key);
      const by = mine ? "" : ` [${esc(c.requestor)}]`;
      const bad = HALLUCINATED.has(c.name) && mine;
      return `<details class="call${dup ? " dup" : ""}${mine ? "" : " theirs"}${bad ? " halluc" : ""}">
        <summary>${esc(c.name)}(${esc(shortArgs(c.arguments))})${by}</summary>
        <pre>${esc(JSON.stringify(c.arguments, null, 2))}</pre></details>`;
    }).join("");
    return `<div class="msg ${esc(m.role)}"><div class="who">${esc(who)}${
      m.turn !== undefined ? " · " + m.turn : ""}</div>${body}${calls}</div>`;
  }).join("");
  return `<h3>trace</h3>${parts}`;
}

function shortArgs(args) {
  if (!args || typeof args !== "object") return "";
  const s = Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  return s.length > 70 ? s.slice(0, 70) + "…" : s;
}

renderTop();
const initial = fromHash();
if (initial) state.sel = initial;
renderLeft();
renderRight();
