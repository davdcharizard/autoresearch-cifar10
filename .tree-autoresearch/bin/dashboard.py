#!/usr/bin/env python3
"""Tree Autoresearch dashboard — passive read-only monitor.

Serves the experiment trees under <project-root>/.tree-autoresearch/ as a
live web page. Never writes, never locks: the loop is completely unaffected
by a running dashboard.

Endpoints (the server does NOT parse TSVs — all parsing is in the browser):
    /            app shell (single self-contained HTML page, D3 v7 via CDN)
    /api/goals   JSON goal slugs grouped by 04-results.tsv availability
    /tsv/{slug}  the goal's 04-results.tsv served verbatim

Usage:
    python3 dashboard.py [--port N] [--host H] [--root PROJECT_ROOT]

Root resolution when --root is omitted: if this script lives at
.tree-autoresearch/bin/dashboard.py the project root is derived from the
script location; otherwise the ancestors of the current directory are
searched for a .tree-autoresearch/ directory.
"""

import argparse
import contextlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def detect_root(cli_root):
    if cli_root:
        root = Path(cli_root).resolve()
        if not (root / ".tree-autoresearch").is_dir():
            sys.exit(
                f"error: {root}/.tree-autoresearch/ not found — pass the "
                "project root that contains .tree-autoresearch/"
            )
        return root
    here = Path(__file__).resolve().parent
    if here.name == "bin" and here.parent.name == ".tree-autoresearch":
        return here.parent.parent
    d = Path.cwd().resolve()
    while True:
        if (d / ".tree-autoresearch").is_dir():
            return d
        if d.parent == d:
            break
        d = d.parent
    sys.exit(
        "error: no .tree-autoresearch/ found in the current directory or its "
        "ancestors — run from inside an initialized project or pass "
        "--root <project-root>"
    )


def make_handler(root, page):
    goals_dir = root / ".tree-autoresearch" / "goals"

    class Handler(BaseHTTPRequestHandler):
        server_version = "TreeAutoresearchDashboard/1.0"

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server API)
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif path == "/api/goals":
                slugs = []
                missing = []
                if goals_dir.is_dir():
                    for d in sorted(goals_dir.iterdir()):
                        if not d.is_dir():
                            continue
                        target = slugs if (d / "04-results.tsv").is_file() else missing
                        target.append(d.name)
                body = json.dumps({"goals": slugs, "missing": missing}).encode()
                self._send(200, body, "application/json")
            elif path.startswith("/tsv/"):
                slug = path[len("/tsv/") :]
                if not SLUG_RE.match(slug):
                    self._send(400, b"bad slug", "text/plain")
                    return
                goal_dir = goals_dir / slug
                if not goal_dir.is_dir():
                    self._send(404, b"no such goal", "text/plain")
                    return
                tsv = goal_dir / "04-results.tsv"
                if not tsv.is_file():
                    self._send(409, b"results TSV missing", "text/plain")
                    return
                try:
                    body = tsv.read_bytes()
                except FileNotFoundError:
                    code = 409 if goal_dir.is_dir() else 404
                    self._send(code, b"results TSV missing", "text/plain")
                    return
                self._send(200, body, "text/tab-separated-values; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):
            pass  # quiet — this is a background monitor

    return Handler


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tree Autoresearch — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
  :root {
    --green:  #4ade80;
    --gray:   #64748b;
    --red:    #f87171;
    --yellow: #facc15;
    --gold:   #fbbf24;
    --bg:     #0b1220;
    --panel:  #101a2e;
    --line:   #1e2a44;
    --ink:    #e2e8f0;
    --muted:  #7c8db0;
    --accent: #38bdf8;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace;
    background: var(--bg); color: var(--ink); display: flex; height: 100vh; overflow: hidden;
    font-size: 13px;
  }
  #sidebar {
    width: 260px; min-width: 260px; background: var(--panel); border-right: 1px solid var(--line);
    padding: 16px 12px; display: flex; flex-direction: column; gap: 4px;
  }
  #sidebar h1 { font-size: 13px; color: var(--accent); letter-spacing: .08em; }
  #sidebar .sub { font-size: 10px; color: var(--muted); margin-bottom: 14px; }
  .goal-item { padding: 9px 10px; border-left: 2px solid transparent; cursor: pointer; }
  .goal-item:hover { background: #16233c; }
  .goal-item.active { background: #16233c; border-left-color: var(--accent); }
  .goal-item .slug { font-size: 12px; color: var(--ink); }
  .goal-item .best { font-size: 10px; color: var(--muted); margin-top: 3px; }
  .goal-item.missing .best { color: var(--red); }
  .goal-item.active .best b { color: var(--gold); }
  #sidebar .foot { margin-top: auto; font-size: 9.5px; color: var(--muted); }
  .live { color: var(--green); }
  .live.down { color: var(--red); }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #topbar {
    display: flex; align-items: baseline; gap: 18px; padding: 12px 20px;
    border-bottom: 1px solid var(--line); background: var(--panel);
  }
  #topbar .goal-title { font-size: 13px; color: var(--ink); flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .stat { font-size: 11px; color: var(--muted); white-space: nowrap; }
  .stat b { color: var(--ink); }
  .stat.best b { color: var(--gold); }
  #viewtoggle { display: flex; gap: 2px; }
  #viewtoggle button {
    border: 1px solid var(--line); background: transparent; color: var(--muted);
    padding: 4px 12px; font-size: 11px; font-family: inherit; cursor: pointer;
  }
  #viewtoggle button.active { border-color: var(--accent); color: var(--accent); }
  #canvas { flex: 1; overflow: hidden; position: relative;
    background-image: radial-gradient(circle, #16233c 1px, transparent 1px);
    background-size: 26px 26px; }
  svg { display: block; }
  #svg.pannable { cursor: grab; }
  #svg.pannable:active { cursor: grabbing; }
  #empty {
    position: absolute; inset: 0; display: none; align-items: center; justify-content: center;
    color: var(--muted); font-size: 12px; text-align: center; line-height: 2;
  }
  #legend {
    position: absolute; right: 16px; top: 12px; background: rgba(16,26,46,.92);
    border: 1px solid var(--line); padding: 9px 12px; font-size: 10.5px; color: var(--muted); line-height: 1.9;
  }
  #legend .sym { display: inline-block; width: 14px; text-align: center; margin-right: 4px; }
  #hints {
    position: absolute; bottom: 12px; left: 0; right: 0; text-align: center;
    font-size: 10.5px; color: var(--muted); pointer-events: none; display: none;
  }
  #fitbtn {
    position: absolute; right: 16px; bottom: 14px; display: none;
    border: 1px solid var(--line); background: rgba(16,26,46,.92); color: var(--muted);
    padding: 5px 12px; font-size: 11px; font-family: inherit; cursor: pointer;
  }
  #fitbtn:hover { border-color: var(--accent); color: var(--accent); }
  #tooltip {
    position: fixed; pointer-events: none; z-index: 10; display: none; max-width: 340px;
    background: #030712; color: var(--ink); border: 1px solid var(--accent);
    padding: 10px 12px; font-size: 11px; line-height: 1.6;
  }
  #tooltip .t-desc { color: var(--accent); margin-bottom: 4px; }
  #tooltip .t-row { color: var(--muted); }
  #tooltip .t-row b { color: var(--ink); }
  #tooltip .t-branch { margin-top: 6px; padding-top: 6px; border-top: 1px dashed var(--line); color: var(--gold); }
  .node-label { font-size: 11px; fill: var(--ink); }
  .node-metric { font-size: 9.5px; fill: var(--muted); }
  .link { fill: none; stroke: #2b3c5e; stroke-width: 1.5px; }
  .link.failed { stroke-dasharray: 3 4; stroke: #22304e; }
  .branch-tag { font-size: 10px; letter-spacing: .06em; }
  .axis text { font-size: 10px; fill: var(--muted); font-family: inherit; }
  .axis line, .axis path { stroke: var(--line); }
  .glow { filter: drop-shadow(0 0 6px rgba(251,191,36,.8)); }
</style>
</head>
<body>

<div id="sidebar">
  <h1>▍TREE-AUTORESEARCH</h1>
  <div class="sub"><span class="live" id="live-dot">●</span> <span id="live-text">live</span></div>
  <div id="goal-list"></div>
  <div class="foot" id="src-line"></div>
</div>

<div id="main">
  <div id="topbar">
    <div class="goal-title" id="goal-title">// waiting for goals…</div>
    <span class="stat" id="chip-metric"></span>
    <span class="stat" id="chip-nodes"></span>
    <span class="stat best" id="chip-best"></span>
    <div id="viewtoggle">
      <button id="btn-tree" class="active">TREE</button>
      <button id="btn-traj">TRAJECTORY</button>
    </div>
  </div>
  <div id="canvas">
    <div id="legend"></div>
    <div id="empty">no goal initialized yet</div>
    <svg id="svg"></svg>
    <div id="hints">drag pan · wheel zoom · double-click fit</div>
    <button id="fitbtn">⌖ FIT</button>
  </div>
</div>

<div id="tooltip"></div>

<script>
/* ============ TSV parsing — one parser, in the browser ============ */
function parseGoalTsv(raw) {
  const lines = raw.trim().split("\n");
  const header = { branches: [] };
  const bodyLines = [];
  for (const ln of lines) {
    if (ln.startsWith("# metric:")) {
      for (const part of ln.slice(1).split("|")) {
        const [k, ...v] = part.split(":");
        header[k.trim()] = v.join(":").trim();
      }
    } else if (ln.startsWith("# goal:")) {
      header.goal = ln.replace("# goal:", "").trim();
    } else if (ln.startsWith("# branch:")) {
      const seg = {};
      for (const part of ln.slice(1).split("|")) {
        const [k, ...v] = part.split(":");
        seg[k.trim()] = v.join(":").trim();
      }
      header.branches.push(seg);
    } else if (!ln.startsWith("#")) {
      bodyLines.push(ln);
    }
  }
  if (!bodyLines.length) return { header, rows: [] };
  // defensively drop a torn trailing line (column-count mismatch)
  const cols = bodyLines[0].split("\t").length;
  const clean = bodyLines.filter((l, i) => i === 0 || l.split("\t").length === cols);
  const rows = d3.tsvParse(clean.join("\n"), d => ({ ...d, metricNum: +d.metric }));
  return { header, rows };
}

const VERDICT_COLOR = {
  baseline: "#f8fafc", improvement: "#4ade80", "no-improvement": "#64748b",
  invalid: "#f87171", crash: "#facc15"
};
/* BASE is stored with verdict 'improvement' (extendable root) — color it as baseline */
const nodeColor = d => d.exp_id === "BASE" ? VERDICT_COLOR.baseline : VERDICT_COLOR[d.verdict];
const POLL_MS = __POLL_MS__;
const BRANCH_COLOR = d3.scaleOrdinal(["#38bdf8", "#c084fc", "#2dd4bf", "#f472b6", "#fb923c"]);

let GOALS = [];
let MISSING = new Set();
const DATA = {};      // slug -> {header, rows}
const RAW = {};       // slug -> raw tsv text (change detection)
const SEEN = {};      // slug -> Set of exp_ids already animated in
const TX = {};        // slug -> d3.zoomTransform of the tree view (survives re-renders)
let fitTree = null;   // set by renderTree; the ⌖ FIT button and double-click call it
let activeGoal = null;
let activeView = "tree";
const tooltip = d3.select("#tooltip");

/* ============ polling ============ */
async function poll() {
  try {
    const res = await fetch("/api/goals");
    if (!res.ok) throw new Error(`goals request failed: ${res.status}`);
    const { goals, missing = [] } = await res.json();
    const nextGoals = [...goals, ...missing].sort();
    const nextMissing = new Set(missing);
    let changed = nextGoals.length !== GOALS.length
      || nextGoals.some((slug, i) => slug !== GOALS[i])
      || missing.length !== MISSING.size
      || missing.some(slug => !MISSING.has(slug));
    GOALS = nextGoals;
    MISSING = nextMissing;
    const vanished = new Set();
    await Promise.all(goals.map(async slug => {
      const tsvRes = await fetch("/tsv/" + slug);
      if (tsvRes.status === 404) {
        vanished.add(slug);
        changed = true;
        return;
      }
      if (tsvRes.status === 409) {
        MISSING.add(slug);
        changed = true;
        return;
      }
      if (!tsvRes.ok) throw new Error(`TSV request failed for ${slug}: ${tsvRes.status}`);
      const raw = await tsvRes.text();
      if (RAW[slug] !== raw) {
        RAW[slug] = raw;
        DATA[slug] = parseGoalTsv(raw);
        changed = true;
      }
    }));
    if (vanished.size) {
      GOALS = GOALS.filter(slug => !vanished.has(slug));
      vanished.forEach(slug => MISSING.delete(slug));
    }
    for (const slug of Object.keys(DATA)) {
      if (!GOALS.includes(slug) || MISSING.has(slug)) {
        delete DATA[slug];
        delete RAW[slug];
      }
    }
    const fallbackGoal = () => GOALS.find(slug => !MISSING.has(slug)) || GOALS[0] || null;
    if (!activeGoal && GOALS.length) { activeGoal = fallbackGoal(); changed = true; }
    if (activeGoal && !GOALS.includes(activeGoal)) { activeGoal = fallbackGoal(); changed = true; }
    d3.select("#live-dot").classed("down", false);
    d3.select("#live-text").text(`live · poll ${POLL_MS / 1000}s`);
    if (changed) render();
  } catch (e) {
    d3.select("#live-dot").classed("down", true);
    d3.select("#live-text").text("server unreachable — retrying");
  }
}

/* ============ sidebar ============ */
function renderSidebar() {
  const list = d3.select("#goal-list").selectAll(".goal-item").data(GOALS, d => d);
  list.exit().remove();
  const enter = list.enter().append("div").attr("class", "goal-item")
    .on("click", (e, d) => { activeGoal = d; render(); });
  enter.append("div").attr("class", "slug");
  enter.append("div").attr("class", "best");
  const all = enter.merge(list);
  all.classed("active", d => d === activeGoal);
  all.classed("missing", d => MISSING.has(d));
  all.select(".slug").text(d => "▸ " + d);
  all.select(".best").html(d => {
    if (MISSING.has(d)) return "04-results.tsv missing";
    const g = DATA[d];
    if (!g || !g.rows.length) return "initializing…";
    const h = g.header;
    return `best <b>${h.best} ${h.unit}</b> @ ${h.best_exp} · ${g.rows.length - 1} exps`;
  });
}

/* ============ topbar ============ */
function renderTopbar() {
  d3.select("#btn-tree").classed("active", activeView === "tree");
  d3.select("#btn-traj").classed("active", activeView === "traj");
  if (activeGoal && MISSING.has(activeGoal)) {
    d3.select("#goal-title").text("// " + activeGoal).attr("title", "results TSV missing");
    ["#chip-metric", "#chip-nodes", "#chip-best"].forEach(s => d3.select(s).html(""));
    return;
  }
  if (!activeGoal || !DATA[activeGoal] || !DATA[activeGoal].rows.length) {
    d3.select("#goal-title").text("// waiting for goals…");
    ["#chip-metric", "#chip-nodes", "#chip-best"].forEach(s => d3.select(s).html(""));
    return;
  }
  const { header, rows } = DATA[activeGoal];
  d3.select("#goal-title").text("// " + activeGoal).attr("title", header.goal || "");
  d3.select("#chip-metric").html(`metric <b>${header.metric}</b> (${header.unit}, ${header.direction} is better)`);
  d3.select("#chip-nodes").html(`exps <b>${rows.length - 1}</b>`);
  d3.select("#chip-best").html(`global best <b>${header.best} ${header.unit}</b> @ ${header.best_exp}`);
}

function renderLegend() {
  // symbols mirror what each view actually draws
  const entries = activeView === "tree"
    ? [
        ["●", VERDICT_COLOR.baseline, "baseline"],
        ["●", VERDICT_COLOR.improvement, "improvement"],
        ["●", VERDICT_COLOR["no-improvement"], "no-improvement"],
        ["●", VERDICT_COLOR.invalid, "invalid"],
        ["●", VERDICT_COLOR.crash, "crash"],
        ["○", "var(--gold)", "global best"],
      ]
    : [
        ["■", VERDICT_COLOR.baseline, "baseline"],
        ["■", VERDICT_COLOR.improvement, "improvement"],
        ["■", VERDICT_COLOR["no-improvement"], "no-improvement"],
        ["✚", VERDICT_COLOR.invalid, "invalid"],
        ["▲", VERDICT_COLOR.crash, "crash"],
        ["○", "var(--gold)", "global best"],
      ];
  d3.select("#legend").html(
    entries.map(([s, c, n]) => `<span class="sym" style="color:${c}">${s}</span>${n}`).join("<br>")
  );
}

function showTip(event, d, header) {
  const b = header.branches.find(x => x.branch === d.data.branch);
  tooltip.style("display", "block")
    .style("left", (event.clientX + 14) + "px").style("top", (event.clientY + 10) + "px")
    .html(`
      <div class="t-desc">[${d.data.exp_id === "BASE" ? "BASE" : "EXP-" + d.data.exp_id}] ${d.data.description}</div>
      <div class="t-row">metric <b>${d.data.metric} ${header.unit || ""}</b> · verdict <b>${d.data.verdict}</b></div>
      <div class="t-row">commit <b>${d.data.commit}</b> · branch <b>${d.data.branch}</b></div>
      ${b ? `<div class="t-branch">» ${b.summary}</div>` : ""}`);
}
function hideTip() { tooltip.style("display", "none"); }

/* new-node animation: scale in anything not seen on a previous render of this goal */
function animateNew(nodeSel, getId) {
  const seen = SEEN[activeGoal] || (SEEN[activeGoal] = new Set());
  const fresh = nodeSel.filter(d => !seen.has(getId(d)));
  fresh.attr("opacity", 0).transition().duration(600).attr("opacity", 1);
  nodeSel.each(d => seen.add(getId(d)));
}

/* ============ tree view (vertical, top → down) ============ */
function renderTree() {
  const { header, rows } = DATA[activeGoal];
  const svg = d3.select("#svg");
  svg.selectAll("*").remove();

  const root = d3.stratify().id(d => d.exp_id).parentId(d => d.parent_id === "-" ? null : d.parent_id)(rows);
  const W = document.getElementById("canvas").clientWidth;
  const H = document.getElementById("canvas").clientHeight;
  svg.attr("width", W).attr("height", H).classed("pannable", true);
  // layout in virtual space sized to the tree; pan/zoom maps it onto the canvas
  const LW = Math.max(W, 160 + root.leaves().length * 130);
  const LH = Math.max(H, 180 + root.height * 120);
  d3.tree().size([LW - 160, LH - 170])(root);
  const g = svg.append("g");

  g.selectAll(".link").data(root.links()).enter().append("path")
    .attr("class", d => "link" + (["invalid", "crash", "no-improvement"].includes(d.target.data.verdict) ? " failed" : ""))
    .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));

  const node = g.selectAll(".node").data(root.descendants()).enter().append("g")
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .on("mousemove", (e, d) => showTip(e, d, header)).on("mouseleave", hideTip);

  node.filter(d => d.data.exp_id === header.best_exp).append("circle")
    .attr("class", "glow").attr("r", 16).attr("fill", "none")
    .attr("stroke", "var(--gold)").attr("stroke-width", 2);

  node.append("rect")
    .attr("x", -9).attr("y", -9).attr("width", 18).attr("height", 18).attr("rx", 4)
    .attr("fill", "#0b1220")
    .attr("stroke", d => nodeColor(d.data)).attr("stroke-width", 2);
  node.append("circle").attr("r", 3.5).attr("fill", d => nodeColor(d.data));

  node.append("text").attr("class", "node-label").attr("dy", -18).attr("text-anchor", "middle")
    .text(d => d.data.exp_id === "BASE" ? "BASE" : d.data.exp_id);
  node.append("text").attr("class", "node-metric").attr("dy", 26).attr("text-anchor", "middle")
    .text(d => isNaN(d.data.metricNum) ? d.data.verdict.toUpperCase() : d3.format(",")(d.data.metricNum));

  const tips = {};
  rows.forEach(r => { if (r.verdict === "improvement" || r.verdict === "baseline") tips[r.branch] = r.exp_id; });
  node.filter(d => tips[d.data.branch] === d.data.exp_id && d.data.exp_id !== "BASE")
    .append("text").attr("class", "branch-tag").attr("dy", 40).attr("text-anchor", "middle")
    .attr("fill", d => BRANCH_COLOR(d.data.branch)).text(d => "⎇ " + d.data.branch);

  animateNew(node, d => d.data.exp_id);

  /* drag to pan, wheel to zoom, double-click to re-fit; view survives re-renders */
  const zoom = d3.zoom().scaleExtent([0.2, 4])
    .on("zoom", e => { g.attr("transform", e.transform); TX[activeGoal] = e.transform; });
  const fit = () => {
    const b = g.node().getBBox();
    const k = Math.min(1, 0.92 * Math.min(W / b.width, H / b.height));
    return d3.zoomIdentity
      .translate((W - b.width * k) / 2 - b.x * k, (H - b.height * k) / 2 - b.y * k)
      .scale(k);
  };
  fitTree = () => svg.transition().duration(350).call(zoom.transform, fit());
  svg.call(zoom).on("dblclick.zoom", null).on("dblclick", () => fitTree());
  svg.call(zoom.transform, TX[activeGoal] || fit());
}

/* ============ trajectory view ============ */
function renderTraj() {
  const { header, rows } = DATA[activeGoal];
  const svg = d3.select("#svg").classed("pannable", false)
    .on(".zoom", null).on("dblclick", null);
  fitTree = null;
  svg.selectAll("*").remove();
  const W = document.getElementById("canvas").clientWidth, H = document.getElementById("canvas").clientHeight;
  svg.attr("width", W).attr("height", H);
  const m = { top: 46, right: 200, bottom: 52, left: 84 };

  const pts = rows.map((r, i) => ({ ...r, step: i }));
  const numeric = pts.filter(p => !isNaN(p.metricNum));
  if (!numeric.length) return;
  const higher = header.direction === "higher";
  let best = numeric[0].metricNum;
  const bestLine = pts.map(p => {
    if (!isNaN(p.metricNum) && (higher ? p.metricNum > best : p.metricNum < best)) best = p.metricNum;
    return { step: p.step, best };
  });

  const x = d3.scaleLinear().domain([0, Math.max(pts.length - 1, 1)]).range([m.left, W - m.right]);
  const ext = d3.extent(numeric, p => p.metricNum), pad = (ext[1] - ext[0]) * 0.15 || 1;
  const y = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([H - m.bottom, m.top]);

  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.bottom})`)
    .call(d3.axisBottom(x).ticks(pts.length).tickFormat(i => pts[i] ? (pts[i].exp_id === "BASE" ? "BASE" : pts[i].exp_id) : ""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(6));
  svg.append("text").attr("x", m.left - 62).attr("y", m.top - 18).attr("font-size", 10.5)
    .attr("fill", "var(--muted)").text(`${header.metric} (${header.unit})`);

  svg.append("path").datum(bestLine).attr("fill", "none")
    .attr("stroke", "var(--gold)").attr("stroke-width", 1.5).attr("stroke-dasharray", "5 3")
    .attr("d", d3.line().x(p => x(p.step)).y(p => y(p.best)).curve(d3.curveStepAfter));

  const failedY = p => {
    const parent = pts.find(q => q.exp_id === p.parent_id);
    return y(parent && !isNaN(parent.metricNum) ? parent.metricNum : ext[0]);
  };
  const gp = svg.selectAll(".pt").data(pts).enter().append("g")
    .attr("transform", p => `translate(${x(p.step)},${isNaN(p.metricNum) ? failedY(p) : y(p.metricNum)})`)
    .on("mousemove", (e, p) => showTip(e, { data: p }, header)).on("mouseleave", hideTip);

  gp.append("path")
    .attr("d", p => d3.symbol()
      .type(p.verdict === "crash" ? d3.symbolTriangle : p.verdict === "invalid" ? d3.symbolCross : d3.symbolSquare)
      .size(p.exp_id === header.best_exp ? 170 : 110)())
    .attr("fill", p => nodeColor(p)).attr("stroke", "#0b1220").attr("stroke-width", 1.5);

  gp.filter(p => p.exp_id === header.best_exp).append("circle")
    .attr("class", "glow").attr("r", 13).attr("fill", "none").attr("stroke", "var(--gold)").attr("stroke-width", 1.5);

  svg.append("text").attr("x", (m.left + W - m.right) / 2).attr("y", H - 14)
    .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", "var(--muted)")
    .text("experiment order →");

  animateNew(gp, p => p.exp_id);
}

function render() {
  renderSidebar(); renderTopbar(); renderLegend();
  d3.select("#src-line").text(activeGoal ? `source: goals/${activeGoal}/04-results.tsv` : "");
  const isMissing = activeGoal && MISSING.has(activeGoal);
  const hasData = activeGoal && !isMissing && DATA[activeGoal] && DATA[activeGoal].rows.length;
  d3.select("#empty").text(
    isMissing ? `goals/${activeGoal}/04-results.tsv is missing` : "no goal initialized yet"
  );
  d3.select("#empty").style("display", hasData ? "none" : "flex");
  const treeCtl = hasData && activeView === "tree";
  d3.select("#fitbtn").style("display", treeCtl ? "block" : "none");
  d3.select("#hints").style("display", treeCtl ? "block" : "none");
  if (!hasData) { d3.select("#svg").selectAll("*").remove(); return; }
  activeView === "tree" ? renderTree() : renderTraj();
}
d3.select("#btn-tree").on("click", () => { activeView = "tree"; render(); });
d3.select("#btn-traj").on("click", () => { activeView = "traj"; render(); });
d3.select("#fitbtn").on("click", () => { if (fitTree) fitTree(); });
window.addEventListener("resize", render);
render();
poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=None, help="project root containing .tree-autoresearch/")
    ap.add_argument("--poll-ms", type=int, default=2000, help="browser poll interval in ms")
    args = ap.parse_args()

    root = detect_root(args.root)
    page = PAGE.replace("__POLL_MS__", str(max(args.poll_ms, 250)))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root, page))
    print(f"tree-autoresearch dashboard: http://{args.host}:{args.port}/")
    print(f"watching: {root / '.tree-autoresearch' / 'goals'} (read-only, poll-based)")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
