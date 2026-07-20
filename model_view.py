"""Standalone family-level model view. Grain-AGNOSTIC: a constraint touches
1..n grains (its application grain + the birth-grains of every variable it
references), and inter-grain links fall out of joins with no special-casing.

Tacked onto to_frames(); not integrated into report.py (that refactor is later).
"""
import json
import re
import polars as pl

_VAR_RE = re.compile(r"var_\d+")


def _constraint_variables(cons):
    """Parse var ids out of each constraint's expr string -> long edge table
    (con_id, var_id). This is the missing relation; here we recover it from
    the formatted expr rather than changing capture."""
    recs = []
    for con_id, expr in zip(cons["con_id"], cons["expr"]):
        for vid in dict.fromkeys(_VAR_RE.findall(expr or "")):  # dedupe, keep order
            recs.append({"con_id": con_id, "var_id": vid})
    if not recs:
        return pl.DataFrame(schema={"con_id": pl.String, "var_id": pl.String})
    return pl.DataFrame(recs)


def _constraint_grains(cons, variables):
    """Every (con_id, grain_id) link a constraint participates in:
    its OWN application grain, UNION the birth-grain of every var it touches.
    This is the grain-agnostic heart — 1 vs n grains is just how many rows."""
    con_vars = _constraint_variables(cons)

    # grains reached THROUGH variables (var -> its birth grain)
    via_vars = (
        con_vars
        .join(variables.select(["var_id", "birth_grain_id"]), on="var_id", how="left")
        .select(["con_id", pl.col("birth_grain_id").alias("grain_id")])
        .drop_nulls("grain_id")
    )
    # the constraint's own application grain
    own = cons.select(["con_id", "grain_id"])

    return pl.concat([own, via_vars]).unique()


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