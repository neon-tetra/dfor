"""Standalone family-level model view. Grain-AGNOSTIC: a constraint touches
1..n grains (its application grain + the birth-grains of every variable it
references), and inter-grain links fall out of joins with no special-casing.

Tacked onto to_frames(); not integrated into report.py (that refactor is later).
"""
import json
from collections import defaultdict

import polars as pl

from model_analysis import constraint_grains as _constraint_grains
from model_analysis import model_graph


def family_view(frames):
    cons = frames["constraints"]
    members = frames["grain_members"]
    variables = frames["variables"]

    # nodes: one per grain, with the entities it spans + its vars
    grain_entities = members.group_by("grain_id").agg(pl.col("entity").alias("spans"))
    grain_vars = (
        variables.group_by("birth_grain_id")
        .agg(pl.col("entity").unique().alias("var_entities"), pl.len().alias("n_vars"))
        .rename({"birth_grain_id": "grain_id"})
    )
    nodes = (grain_entities.join(grain_vars, on="grain_id", how="left")
             .sort("grain_id"))

    # per-constraint grain participation (1..n grains each)
    con_grains = _constraint_grains(cons, variables)

    # roll grains up to the family, so a family lists ALL grains it connects
    con_family = cons.select(["con_id", "type", "grain_id"]).rename(
        {"grain_id": "app_grain"})
    family_grains = (
        con_grains
        .join(con_family, on="con_id", how="left")
        .group_by(["type", "app_grain"])
        .agg(pl.col("grain_id").unique().alias("connects_grains"))
    )

    # family metadata (count / examples), joined to the grain wiring
    family_meta = (
        cons.group_by(["type", "grain_id"])
        .agg(
            pl.len().alias("count"),
            pl.col("entities").first().alias("entities"),
            pl.col("expr").head(2).alias("examples"),
        )
        .rename({"grain_id": "app_grain"})
    )
    edges = (
        family_meta.join(family_grains, on=["type", "app_grain"], how="left")
        .sort("count", descending=True)
    )

    return {"nodes": nodes.to_dicts(), "edges": edges.to_dicts()}


def to_json(frames, path=None):
    blob = json.dumps(family_view(frames), indent=2, default=str)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(blob)
    return blob

# ---- rough HTML card renderer (throwaway; the JSON is the real artifact) ----
def to_html(frames, path):
    view = family_view(frames)

    def node_card(n):
        spans = ", ".join(n.get("spans") or [])
        vents = ", ".join(n.get("var_entities") or [])
        return f"""
        <div class="node">
          <div class="title">{n['grain_id']}</div>
          <div class="row"><b>spans:</b> {spans}</div>
          <div class="row"><b>vars:</b> {vents} ({n.get('n_vars', 0)})</div>
        </div>"""

    def edge_card(e):
        ents = ", ".join(e.get("entities") or [])
        connects = " → ".join(e.get("connects_grains") or [])   # the multi-grain span
        exs = "<br>".join(e.get("examples") or [])
        bridge = "bridge" if len(e.get("connects_grains") or []) > 1 else "same-grain"
        return f"""
        <div class="edge {bridge}">
          <div class="title">{e['type']} @ {e['app_grain']} &times;{e['count']}</div>
          <div class="row"><b>connects:</b> {connects}</div>
          <div class="row"><b>touches:</b> {ents}</div>
          <div class="row example">{exs}</div>
        </div>"""

    nodes_html = "".join(node_card(n) for n in view["nodes"])
    edges_html = "".join(edge_card(e) for e in view["edges"])

    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body {{ font-family: ui-monospace, monospace; background:#0d1117; color:#c9d1d9; padding:20px; }}
      h2 {{ color:#58a6ff; }}
      .node, .edge {{ border:1px solid #30363d; border-radius:8px; padding:10px 14px;
                      margin:8px 0; background:#161b22; }}
      .edge {{ border-left:3px solid #58a6ff; }}
      .edge.same-grain {{ border-left:3px solid #d29922; }}   /* flag the exception case */
      .node {{ border-left:3px solid #3fb950; }}
      .title {{ font-weight:700; color:#e6edf3; margin-bottom:6px; }}
      .row {{ font-size:13px; margin:2px 0; }}
      .example {{ color:#8b949e; white-space:pre-wrap; }}
      .cols {{ display:grid; grid-template-columns:1fr 2fr; gap:20px; }}
    </style></head><body>
      <div class="cols">
        <div><h2>grains (nodes)</h2>{nodes_html}</div>
        <div><h2>families (edges)</h2>{edges_html}</div>
      </div>
    </body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ---- Mermaid flowchart renderer: pure render layer over model_graph's IR.
# No structural logic here -- swap this for erDiagram (or anything else)
# by writing a new function that consumes the same {"nodes","edges"} shape.
_NODE_SHAPES = {
    "entity": ('["', '"]'),
    "entity_ref": ('["', '"]'),
    "grain": ('(["', '"])'),
    "constraint": ('{{"', '"}}'),
}


def _mmd_escape(label):
    return str(label).replace('"', "'")


def to_mermaid(frames, path=None):
    """Entities/grains nested in per-column subgraphs mirroring
    cardinality_hierarchy; constraint families are their own nodes with
    solid edges to their home grain's neighborhood and dashed 'touches'
    edges fanning out to every other grain they reach -- so a constraint
    spanning 3+ grains never gets squeezed into a single binary line."""
    graph = model_graph(frames)
    nodes, edges = graph["nodes"], graph["edges"]

    lines = ["flowchart TD"]

    columns = sorted(c for c in nodes["column"].unique().to_list() if c is not None)
    for col in columns:
        col_nodes = nodes.filter(pl.col("column") == col)
        roots = col_nodes.filter(
            (pl.col("node_type") == "entity") & (pl.col("parent_id").is_null()))
        title = _mmd_escape(" / ".join(roots["label"].to_list()) or f"column {col}")
        lines.append(f'  subgraph col{col} ["{title}"]')
        for n in col_nodes.iter_rows(named=True):
            open_, close_ = _NODE_SHAPES[n["node_type"]]
            lines.append(f'    {n["node_id"]}{open_}{_mmd_escape(n["label"])}{close_}')
        lines.append("  end")

    for n in nodes.filter(pl.col("column").is_null()).iter_rows(named=True):
        open_, close_ = _NODE_SHAPES[n["node_type"]]
        lines.append(f'  {n["node_id"]}{open_}{_mmd_escape(n["label"])}{close_}')

    for e in edges.iter_rows(named=True):
        arrow = "-->" if e["edge_type"] == "hierarchy" else "-.->"
        label = f'|{_mmd_escape(e["label"])}|' if e["label"] else ""
        lines.append(f'  {e["source"]} {arrow}{label} {e["target"]}')

    lines += [
        "  classDef entity fill:#161b22,stroke:#3fb950,color:#e6edf3;",
        "  classDef entity_ref fill:#161b22,stroke:#3fb950,color:#e6edf3,stroke-dasharray: 3 2;",
        "  classDef grain fill:#161b22,stroke:#58a6ff,color:#e6edf3;",
        "  classDef constraint fill:#161b22,stroke:#d29922,color:#e6edf3;",
    ]
    for node_type in ("entity", "entity_ref", "grain", "constraint"):
        ids = nodes.filter(pl.col("node_type") == node_type)["node_id"].to_list()
        if ids:
            lines.append(f"  class {','.join(ids)} {node_type};")

    text = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def to_mermaid_html(frames, path):
    """Standalone page wrapping to_mermaid()'s output: loads Mermaid from a
    public CDN (a static JS file, not a signup-required service -- no
    account, no Mermaid Chart cloud) and renders on load. Otherwise fully
    self-contained; just open the file in a browser, no server needed."""
    diagram = to_mermaid(frames)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body {{ background:#0d1117; color:#e6edf3; font-family: ui-monospace, monospace; padding:20px; }}
    </style></head><body>
      <pre class="mermaid">
{diagram}
      </pre>
      <script type="module">
        import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
        mermaid.initialize({{ startOnLoad: true, theme: "dark" }});
      </script>
    </body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ---- indented-outline renderer: another pure render layer over model_graph's
# IR, nothing structural here either. One continuous tree (model_graph's
# "column" still orders rows, it just isn't boxed apart visually), grains
# and their constraints inline, and constraint bridges to OTHER grains drawn
# as curved connector lines computed client-side from actual rendered row
# positions (robust to text wrapping / variable row heights, unlike
# precomputing pixel offsets in Python).
def _tree_escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _truncate(text, limit=300):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


_GUIDE_CONT = "│   "   # │
_GUIDE_BLANK = "    "
_GUIDE_TEE = "├── "   # ├──
_GUIDE_ELBOW = "└── "  # └──

# cycled across constraint families so adjacent cards/lines are visually
# distinguishable; kept clear of the green/blue already used for entity/grain
_CONSTRAINT_PALETTE = ["#e3b341", "#bc8cff", "#39c5cf", "#f778ba", "#ffa657"]


def _color_for(cf_id):
    idx = int(cf_id[2:]) % len(_CONSTRAINT_PALETTE)
    return idx, _CONSTRAINT_PALETTE[idx]


def to_tree_html(frames, path):
    graph = model_graph(frames)
    nodes, edges = graph["nodes"], graph["edges"]
    variables = frames["variables"]

    vars_by_grain = {
        r["birth_grain_id"]: (r["var_entities"], r["n_vars"])
        for r in (
            variables.group_by("birth_grain_id")
            .agg(pl.col("entity").unique().alias("var_entities"), pl.len().alias("n_vars"))
            .iter_rows(named=True)
        )
    }

    constraints_by_home = defaultdict(list)
    for n in nodes.filter(pl.col("node_type") == "constraint").iter_rows(named=True):
        constraints_by_home[n["parent_id"]].append(n)

    touches = defaultdict(list)
    for e in edges.filter(pl.col("edge_type") == "touches").iter_rows(named=True):
        touches[e["source"]].append(e["target"])

    tree_nodes = (
        nodes.filter(pl.col("node_type").is_in(["entity", "entity_ref", "grain"]))
        .sort(["column", "row"])
        .to_dicts()
    )
    node_by_id = {n["node_id"]: n for n in tree_nodes}

    # immediate children in display order, keyed by parent_id (None = the
    # top-level peers -- treating "no separate columns" literally, multiple
    # root clusters are just siblings at the top of one tree, not walled off)
    children_of = defaultdict(list)
    for n in tree_nodes:
        children_of[n["parent_id"]].append(n["node_id"])

    def is_last(node_id):
        parent = node_by_id[node_id]["parent_id"]
        return children_of[parent][-1] == node_id

    def guide_for(n):
        # walk the parent chain root-first: a "|" continues under any
        # ancestor that still has siblings coming after it, blank otherwise
        chain = []
        cur_parent = n["parent_id"]
        while cur_parent is not None:
            chain.append(cur_parent)
            cur_parent = node_by_id[cur_parent]["parent_id"]
        chain.reverse()
        prefix = "".join(_GUIDE_BLANK if is_last(a) else _GUIDE_CONT for a in chain)
        return prefix + (_GUIDE_ELBOW if is_last(n["node_id"]) else _GUIDE_TEE)

    connectors = []
    row_html = []
    for n in tree_nodes:
        is_grain = n["node_type"] == "grain"
        guide = _tree_escape(guide_for(n))
        label = f'grain ({_tree_escape(n["label"])})' if is_grain else _tree_escape(n["label"])

        vars_cell = ""
        if is_grain:
            var_entities, n_vars = vars_by_grain.get(n["grain_id"], ([], 0))
            if n_vars:
                vars_cell = f'{_tree_escape(", ".join(var_entities))} ({n_vars})'

        cons_parts = []
        for c in constraints_by_home.get(n["node_id"], []):
            idx, color = _color_for(c["node_id"])
            cons_parts.append(f"""
              <div class="cons" id="{c['node_id']}" style="border-color:{color};">
                <div class="cons-title" style="color:{color};">{_tree_escape(c['label'])}</div>
                <div class="expr fields">{_tree_escape(_truncate(c['example_fields']))}</div>
                <div class="expr">{_tree_escape(_truncate(c['example_ids']))}</div>
              </div>""")
            # a line to its OWN home grain too (always horizontal, same row)
            # -- redundant with position alone, but makes "this constraint
            # lives here" explicit and consistent with every other line
            connectors.append({
                "from": c["node_id"], "to": f"gutter-{n['node_id']}",
                "color": color, "marker": f"arrow-{idx}",
            })
            for target in touches.get(c["node_id"], []):
                # anchor to the gutter's own edges, not the row's full-width
                # edge, so the line only ever crosses the gutter -- never
                # the tree/vars content itself
                connectors.append({
                    "from": c["node_id"], "to": f"gutter-{target}",
                    "color": color, "marker": f"arrow-{idx}",
                })

        row_html.append(f"""
        <tr id="row-{n['node_id']}" class="{n['node_type']}">
          <td class="tree"><span class="guide">{guide}</span><span class="label">{label}</span></td>
          <td class="vars">{vars_cell}</td>
          <td class="gutter" id="gutter-{n['node_id']}"></td>
          <td class="cons-col">{"".join(cons_parts)}</td>
        </tr>""")

    connectors_json = json.dumps(connectors)
    markers_svg = "".join(
        f'<marker id="arrow-{i}" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{color}"></path></marker>'
        for i, color in enumerate(_CONSTRAINT_PALETTE)
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body {{ font-family: ui-monospace, monospace; background:#0d1117; color:#e6edf3; padding:20px; }}
      h2 {{ color:#58a6ff; }}
      table {{ border-collapse: separate; border-spacing: 0 5px; width: 100%; table-layout: fixed; }}
      td {{ vertical-align: top; padding: 8px 12px; }}
      tr.entity {{ background: #142b1a; }}
      tr.grain {{ background: #122642; }}
      tr.entity_ref {{ background: #10151c; }}
      tr.entity > td.tree {{ border-left: 5px solid #3fb950; }}
      tr.grain > td.tree {{ border-left: 5px solid #58a6ff; }}
      tr.entity_ref > td.tree {{ border-left: 5px dashed #3a4552; }}
      tr.entity > td.tree .label {{ color:#56d364; font-weight: 700; }}
      tr.grain > td.tree .label {{ color:#79c0ff; font-weight: 700; }}
      tr.entity_ref > td.tree .label {{ color:#7d8798; font-style: italic; }}
      .guide {{ color:#6e7b8c; white-space: pre; }}
      td.vars {{ color:#b8c4d1; font-size: 13px; word-break: break-word; }}
      td.gutter {{ padding: 0; border-left: 1px dashed #30363d;
                   border-right: 1px dashed #30363d; }}
      .cons {{ display:block; border:1px solid #e3b341; border-radius:6px;
               padding:8px 12px; margin:4px 0; background:#2b2008; }}
      .cons-title {{ color:#f0c674; font-weight:700; font-size:13px; margin-bottom:3px; }}
      .expr {{ color:#c9d1d9; font-size:12px; white-space:pre-wrap; }}
      .expr.fields {{ color:#f7f0dd; }}
      #tree-container {{ position: relative; }}
      #connector-svg {{ position: absolute; top:0; left:0; pointer-events:none; }}
    </style></head><body>
      <h2>model tree</h2>
      <div id="tree-container">
        <svg id="connector-svg"><defs>{markers_svg}</defs></svg>
        <table>
          <colgroup>
            <col style="width:30%"><col style="width:30%">
            <col style="width:10%"><col style="width:30%">
          </colgroup>
          <tbody>{"".join(row_html)}</tbody>
        </table>
      </div>
      <script>
        const connectors = {connectors_json};
        function draw() {{
          const container = document.getElementById("tree-container");
          const svg = document.getElementById("connector-svg");
          const cRect = container.getBoundingClientRect();
          svg.setAttribute("width", cRect.width);
          svg.setAttribute("height", cRect.height);
          svg.innerHTML = svg.innerHTML.split("</defs>")[0] + "</defs>";
          for (const c of connectors) {{
            const a = document.getElementById(c.from);
            const b = document.getElementById(c.to);
            if (!a || !b) continue;
            const ar = a.getBoundingClientRect(), br = b.getBoundingClientRect();
            // source: LEFT edge of the constraint card (the gutter's right
            // boundary); target: LEFT edge of the target row's gutter cell
            // (the gutter's left boundary) -- opposite sides, so the line
            // actually spans the gutter's width instead of both endpoints
            // landing on the same edge. Same row (own home grain) means
            // y1 == y2, which collapses the curve to a straight horizontal
            // line automatically.
            const x1 = ar.left - cRect.left, y1 = ar.top + ar.height / 2 - cRect.top;
            const x2 = br.left - cRect.left, y2 = br.top + br.height / 2 - cRect.top;
            const mx = (x1 + x2) / 2;
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", `M ${{x1}} ${{y1}} C ${{mx}} ${{y1}}, ${{mx}} ${{y2}}, ${{x2}} ${{y2}}`);
            path.setAttribute("stroke", c.color);
            path.setAttribute("fill", "none");
            path.setAttribute("stroke-width", "2.5");
            path.setAttribute("marker-end", `url(#${{c.marker}})`);
            svg.appendChild(path);
          }}
        }}
        window.addEventListener("load", draw);
        window.addEventListener("resize", draw);
      </script>
    </body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path