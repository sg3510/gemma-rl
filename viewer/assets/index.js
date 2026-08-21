"use strict";

const D = JSON.parse(document.getElementById("payload").textContent);
const RUNS = D.runs;
const state = { a: null, b: null };

const esc = (s) =>
  String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const pct = (v) => (v === null || v === undefined ? "n/a" : (v * 100).toFixed(1) + "%");
const signed = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + " pts";

// Any of these differing between two arms means part of the delta is a change
// of serving stack, not a change of policy — the lesson of D13/D14.
const STACK_KEYS = ["agent_model", "user_model", "max_steps"];

function runsTable() {
  const rows = RUNS.map((r) => {
    const s = r.summary;
    const splits = Object.entries(r.splits)
      .map(([k, n]) => `<span class="badge split-${esc(k)}">${esc(k)} ×${n}</span>`).join(" ");
    return `<tr>
      <td><a href="${esc(r.href)}">${esc(r.name)}</a>
        <div class="sub">${esc(r.family)} / ${esc(r.domain)}</div></td>
      <td>${splits}</td>
      <td class="num">${s.num_tasks} × ${r.trials_planned}</td>
      <td class="num">${s.scored} / ${s.cells}</td>
      <td class="num">${s.errors ? `<b>${s.errors}</b>` : 0} / ${s.missing}</td>
      <td class="num">${pct(s.pass1_macro)}</td>
      <td class="num">${pct(s.pass1_micro)}</td>
      <td class="num">${s.frontier_tasks} (${pct(s.frontier_share)})</td>
    </tr>`;
  }).join("");
  return `<table>
    <tr><th>run</th><th>splits</th><th class="num">tasks × trials</th>
        <th class="num">scored</th><th class="num">err / missing</th>
        <th class="num">pass@1 macro</th><th class="num">pass@1 micro</th>
        <th class="num">frontier</th></tr>${rows}</table>
    <p class="legend">macro = mean of per-task rates (what the paired analysis uses);
    micro = successes / scored rollouts. They differ whenever tasks have unequal
    trial counts.</p>`;
}

function pairPanel() {
  const opts = (cur) => ["none", ...RUNS.map((r) => r.name)]
    .map((n) => `<option${n === cur ? " selected" : ""}>${esc(n)}</option>`).join("");
  const a = RUNS.find((r) => r.name === state.a);
  const b = RUNS.find((r) => r.name === state.b);
  return `<h2>Paired comparison</h2>
    <p class="sub">Per-task deltas on the tasks both runs contain — the test
    <code>reports/eval_protocol.md</code> commits us to. Pooled totals are not it.</p>
    <div class="controls">
      <label>before <select id="sel-a">${opts(state.a || "none")}</select></label>
      <label>after <select id="sel-b">${opts(state.b || "none")}</select></label>
    </div>
    ${a && b ? pairBody(a, b) : `<p class="empty">Pick two runs.</p>`}`;
}

function pairBody(a, b) {
  if (a.name === b.name) return `<p class="empty">Pick two different runs.</p>`;
  const shared = Object.keys(a.tasks).filter((id) => id in b.tasks);
  if (!shared.length)
    return `<p class="empty">These runs share no tasks, so no paired test is possible.</p>`;

  const mismatches = STACK_KEYS
    .filter((k) => JSON.stringify(a.stack[k]) !== JSON.stringify(b.stack[k]))
    .map((k) => `${k}: ${esc(a.stack[k])} → ${esc(b.stack[k])}`);

  const rows = [];
  let sum = 0, scored = 0, better = 0, worse = 0;
  for (const id of shared) {
    const x = a.tasks[id], y = b.tasks[id];
    const usable = x.n > 0 && y.n > 0;
    const delta = usable ? y.pass1 - x.pass1 : null;
    if (usable) {
      sum += delta; scored += 1;
      if (delta > 0) better += 1;
      if (delta < 0) worse += 1;
    }
    rows.push({ id, short: x.short || id, x, y, delta });
  }
  rows.sort((p, q) => (q.delta === null ? -1 : q.delta) - (p.delta === null ? -1 : p.delta));

  const body = rows.map((r) => `<tr>
      <td class="short">${esc(r.short)}</td>
      <td class="num">${r.x.n ? r.x.c + "/" + r.x.n : "—"}</td>
      <td class="num">${r.y.n ? r.y.c + "/" + r.y.n : "—"}</td>
      <td class="num">${r.delta === null ? "—" : signed(r.delta)}</td>
    </tr>`).join("");

  return `
    ${mismatches.length ? `<p class="note">The serving stacks differ
      (${mismatches.join("; ")}). Part of any delta is a change of server, not of
      policy — treat this as diagnostic, not as a before/after result.</p>` : ""}
    <div class="stats">
      <div class="stat"><div class="n">${signed(scored ? sum / scored : 0)}</div>
        <div class="l">mean paired delta</div></div>
      <div class="stat"><div class="n">${scored} / ${shared.length}</div>
        <div class="l">tasks comparable</div></div>
      <div class="stat"><div class="n">${better} ↑ / ${worse} ↓</div>
        <div class="l">tasks moved</div></div>
    </div>
    <p class="legend">A task is comparable only when both runs scored it at least
    once; cells lost to infrastructure errors drop out of the pairing rather than
    counting as failures. This is a point estimate with no confidence interval —
    89% power for +10 points assumes 40 tasks × 8 trials per arm.</p>
    <table><tr><th>task</th><th class="num">${esc(a.name)}</th>
      <th class="num">${esc(b.name)}</th><th class="num">delta</th></tr>${body}</table>`;
}

function render() {
  document.getElementById("body").innerHTML =
    `<h2>Runs</h2>${runsTable()}${pairPanel()}`;
  const sa = document.getElementById("sel-a"), sb = document.getElementById("sel-b");
  sa.onchange = () => { state.a = sa.value === "none" ? null : sa.value; render(); };
  sb.onchange = () => { state.b = sb.value === "none" ? null : sb.value; render(); };
}

render();
