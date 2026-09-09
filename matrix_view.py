"""Bus-matrix model view: grains as rows, entities as columns, a mark where
a grain includes that entity. Landing page links to one static drilldown page
per grain, generated in the same pass. Pure static HTML, no JS.
"""
import os

import polars as pl

from model_analysis import constraint_grains as _constraint_grains
from model_analysis import _VAR_RE

_PALETTE = ["#e3b341", "#bc8cff", "#39c5cf", "#f778ba", "#ffa657"]

_CSS = """
      body { font-family: ui-monospace, monospace; background:#0d1117; color:#e6edf3; padding:20px; }
      h2 { color:#58a6ff; }
      h3 { color:#79c0ff; margin-top:28px; }
      a { color:#58a6ff; text-decoration:none; }
      a:hover { text-decoration:underline; }
      table.matrix { border-collapse: collapse; }
      table.matrix th, table.matrix td { padding: 6px 10px; border: 1px solid #21262d; }
      table.matrix th { color:#8b949e; font-weight:400; }
      table.matrix th.entity { writing-mode: vertical-rl; transform: rotate(180deg);
                               vertical-align: bottom; color:#56d364; font-weight:700; }
      table.matrix td.grain-name { color:#79c0ff; font-weight:700; white-space:nowrap; }
      table.matrix td.dot { text-align:center; }
      table.matrix td.dot.on { background:#122642; color:#79c0ff; font-weight:700; }
      table.matrix td.count { color:#8b949e; text-align:right; }
      table.matrix td.vars { color:#b8c4d1; font-size:12px; max-width:340px; }
      table.matrix td.cons { font-size:12px; max-width:420px; }
      table.matrix tr:hover td { background:#161b22; }
      table.matrix tr:hover td.dot.on { background:#1a3a63; }
      .chip { display:inline-block; border:1px solid; border-radius:10px;
              padding:1px 8px; margin:2px 2px; font-size:12px; }
      .cross { color:#8b949e; font-size:11px; }
      .cons-card { border:1px solid; border-radius:6px; padding:8px 12px;
                   margin:6px 0; background:#161b22; }
      .cons-title { font-weight:700; font-size:13px; margin-bottom:3px; }
      .expr { color:#c9d1d9; font-size:12px; white-space:pre-wrap; }
      .expr.fields { color:#f7f0dd; }
      .meta { color:#8b949e; font-size:13px; margin:4px 0; }
"""


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _page(title, body):
    return (f"<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<style>{_CSS}</style></head><body>{body}</body></html>")


def _matrix_data(frames):
    """Everything both pages need, computed once."""
    members = frames["grain_members"]
    variables = frames["variables"]
    cons = frames["constraints"]
    labels = dict(zip(frames["grain_labels"]["grain_id"],
                      frames["grain_labels"]["label"])) if "grain_labels" in frames else {}

    # grains in declared order; entities in first-appearance order
    grain_order, grain_ents, entity_order = [], {}, []
    for gid, ent in members.iter_rows():
        if gid not in grain_ents:
            grain_ents[gid] = []
            grain_order.append(gid)
        grain_ents[gid].append(ent)
        if ent not in entity_order:
            entity_order.append(ent)

    vars_by_grain = {
        r["birth_grain_id"]: (r["var_entities"], r["n_vars"])
        for r in (variables.group_by("birth_grain_id")
                  .agg(pl.col("entity").unique(maintain_order=True).alias("var_entities"),
                       pl.len().alias("n_vars"))
                  .iter_rows(named=True))
    }

    # constraint families: one per pipe-call site, with user names when given
    var_to_entity = dict(zip(variables["var_id"], variables["entity"]))
    con_grains = _constraint_grains(cons, variables)
    has_name = "name" in cons.columns
    fam_meta = (
        cons.group_by(["type", "grain_id", "call_id"])
        .agg(pl.len().alias("count"),
             pl.col("expr").first().alias("example_ids"),
             *([pl.col("name").first().alias("user_name")] if has_name else []))
        .rename({"grain_id": "app_grain"}))
    fam_grains = (
        con_grains.join(cons.select(["con_id", "type", "grain_id", "call_id"])
                        .rename({"grain_id": "app_grain"}), on="con_id", how="left")
        .group_by(["type", "app_grain", "call_id"])
        .agg(pl.col("grain_id").unique().alias("connects")))
    families = (fam_meta.join(fam_grains, on=["type", "app_grain", "call_id"], how="left")
                .sort(["type", "app_grain", "call_id"]))

    fams = []
    for i, f in enumerate(families.iter_rows(named=True)):
        example_ids = f["example_ids"] or ""
        fams.append({
            "cf_id": f"cf{i}",
            "color": _PALETTE[i % len(_PALETTE)],
            "app_grain": f["app_grain"],
            "title": (f.get("user_name") or f["type"]),
            "verb": f["type"],
            "count": f["count"],
            "example_ids": example_ids,
            "example_fields": _VAR_RE.sub(
                lambda m: var_to_entity.get(m.group(0), m.group(0)), example_ids),
            "connects": [g for g in (f["connects"] or []) if g != f["app_grain"]],
        })

    def display(gid):
        return labels.get(gid, gid)

    return {
        "grain_order": grain_order, "grain_ents": grain_ents,
        "entity_order": entity_order, "vars_by_grain": vars_by_grain,
        "families": fams, "display": display,
    }


def to_matrix_html(frames, path):
    """Landing page: the bus matrix. Also writes one drilldown page per grain
    (grain_<id>.html) into the same directory."""
    d = _matrix_data(frames)
    out_dir = os.path.dirname(os.path.abspath(path))

    header = ("<tr><th></th>"
              + "".join(f'<th class="entity">{_esc(e)}</th>' for e in d["entity_order"])
              + "<th>vars</th><th>constraints</th></tr>")

    rows = []
    for gid in d["grain_order"]:
        ents = set(d["grain_ents"][gid])
        var_entities, n_vars = d["vars_by_grain"].get(gid, ([], 0))
        dots = "".join(
            f'<td class="dot on">&#9679;</td>' if e in ents else '<td class="dot"></td>'
            for e in d["entity_order"])
        vars_cell = f'{_esc(", ".join(var_entities))} ({n_vars})' if n_vars else ""
        chips = ""
        for f in d["families"]:
            if f["app_grain"] != gid:
                continue
            cross = "".join(f' <span class="cross">&#8599;{_esc(d["display"](g))}</span>'
                            for g in f["connects"])
            chips += (f'<span class="chip" style="border-color:{f["color"]};'
                      f'color:{f["color"]};">{_esc(f["title"])} &times;{f["count"]}</span>{cross}')
        rows.append(
            f'<tr><td class="grain-name"><a href="grain_{gid}.html">{_esc(d["display"](gid))}</a></td>'
            f'{dots}<td class="vars">{vars_cell}</td><td class="cons">{chips}</td></tr>')

    body = (f"<h2>model matrix</h2>"
            f'<table class="matrix">{header}{"".join(rows)}</table>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_page("model matrix", body))

    for gid in d["grain_order"]:
        _write_grain_page(d, gid, os.path.join(out_dir, f"grain_{gid}.html"),
                          os.path.basename(path))
    return path


def _write_grain_page(d, gid, path, index_name):
    var_entities, n_vars = d["vars_by_grain"].get(gid, ([], 0))
    ents = d["grain_ents"][gid]

    cards = []
    for f in d["families"]:
        if f["app_grain"] != gid and gid not in f["connects"]:
            continue
        home = ("" if f["app_grain"] == gid else
                f' <span class="cross">(home: <a href="grain_{f["app_grain"]}.html">'
                f'{_esc(d["display"](f["app_grain"]))}</a>)</span>')
        cross = "".join(
            f' <span class="cross">&#8599;<a href="grain_{g}.html">{_esc(d["display"](g))}</a></span>'
            for g in f["connects"] if g != gid)
        title = _esc(f["title"])
        if f["title"] != f["verb"]:
            title += f' <span class="cross">[{_esc(f["verb"])}]</span>'
        cards.append(f"""
          <div class="cons-card" style="border-color:{f['color']};">
            <div class="cons-title" style="color:{f['color']};">{title} &times;{f['count']}{home}{cross}</div>
            <div class="expr fields">{_esc(f['example_fields'])}</div>
            <div class="expr">{_esc(f['example_ids'])}</div>
          </div>""")

    body = (f'<div class="meta"><a href="{index_name}">&larr; matrix</a></div>'
            f"<h2>{_esc(d['display'](gid))}</h2>"
            f'<div class="meta">{_esc(gid)} &mdash; {_esc(" x ".join(ents))}</div>'
            f'<div class="meta">vars: {_esc(", ".join(var_entities))} ({n_vars})</div>'
            f"<h3>constraints</h3>{''.join(cards) or '<div class=\"meta\">none</div>'}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_page(gid, body))
    return path
