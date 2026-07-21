"""Structural analysis over captured diagnostic frames: cardinality + FD
analysis of grains, building an IR of how the model's entities relate.
"""
import re
from collections import defaultdict

import polars as pl

_VAR_RE = re.compile(r"var_\d+")


def constraint_variables(cons):
    """Parse var ids out of each constraint's expr string -> long edge table
    (con_id, var_id). This is the missing relation; recovered from the
    formatted expr rather than threading a var list through capture."""
    recs = []
    for con_id, expr in zip(cons["con_id"], cons["expr"]):
        for vid in dict.fromkeys(_VAR_RE.findall(expr or "")):  # dedupe, keep order
            recs.append({"con_id": con_id, "var_id": vid})
    if not recs:
        return pl.DataFrame(schema={"con_id": pl.String, "var_id": pl.String})
    return pl.DataFrame(recs)


def constraint_grains(cons, variables):
    """Every (con_id, grain_id) link a constraint participates in: its OWN
    application grain, UNION the birth-grain of every var it touches."""
    con_vars = constraint_variables(cons)
    via_vars = (
        con_vars
        .join(variables.select(["var_id", "birth_grain_id"]), on="var_id", how="left")
        .select(["con_id", pl.col("birth_grain_id").alias("grain_id")])
        .drop_nulls("grain_id")
    )
    own = cons.select(["con_id", "grain_id"])
    return pl.concat([own, via_vars]).unique()


def entity_pair_cardinality(frames):
    """One row per ordered pair of entities that co-occur in some grain,
    classified by functional dependency: 1:1, n:1, 1:n, or n:n.

    Candidate pairs come from grain_members (schema-level: which entities
    share a grain). The actual values used to test FD come from
    constraint_rows, since that's the only frame holding real co-occurring
    id/scalar values (captured as the full grain of each add* call — see
    problem.py:_make_constraint_verb). Pairs whose entities never appear
    together in a constraint (vars born but never constrained) come back
    with relationship=None: no evidence either way, not a guess.
    """
    grain_members = frames["grain_members"]
    constraint_rows = frames["constraint_rows"]

    # candidate pairs: any two distinct entities that share a grain
    pairs = (
        grain_members.join(grain_members, on="grain_id", suffix="_rhs")
        .filter(pl.col("entity") != pl.col("entity_rhs"))
        .select(pl.col("entity").alias("lhs"), pl.col("entity_rhs").alias("rhs"))
        .unique()
    )

    recs = []
    for lhs, rhs in pairs.iter_rows():
        left = constraint_rows.filter(pl.col("key") == lhs).select(
            "con_id", pl.col("value").alias("lhs_value"))
        right = constraint_rows.filter(pl.col("key") == rhs).select(
            "con_id", pl.col("value").alias("rhs_value"))
        joined = left.join(right, on="con_id", how="inner").unique(
            ["lhs_value", "rhs_value"])

        if joined.is_empty():
            recs.append({
                "lhs": lhs, "rhs": rhs, "n_pairs": 0,
                "lhs_determines_rhs": None, "rhs_determines_lhs": None,
                "relationship": None,
            })
            continue

        max_rhs_per_lhs = (
            joined.group_by("lhs_value").agg(pl.col("rhs_value").n_unique())
            ["rhs_value"].max())
        max_lhs_per_rhs = (
            joined.group_by("rhs_value").agg(pl.col("lhs_value").n_unique())
            ["lhs_value"].max())
        lhs_determines_rhs = max_rhs_per_lhs == 1
        rhs_determines_lhs = max_lhs_per_rhs == 1

        if lhs_determines_rhs and rhs_determines_lhs:
            relationship = "1:1"
        elif lhs_determines_rhs:
            relationship = "n:1"
        elif rhs_determines_lhs:
            relationship = "1:n"
        else:
            relationship = "n:n"

        recs.append({
            "lhs": lhs, "rhs": rhs, "n_pairs": joined.height,
            "lhs_determines_rhs": lhs_determines_rhs,
            "rhs_determines_lhs": rhs_determines_lhs,
            "relationship": relationship,
        })

    if not recs:
        # zero candidate pairs -- e.g. a reduced core with only one
        # surviving entity. Same shape as the populated case, just empty,
        # so downstream .filter()/.join() calls don't blow up on a
        # columnless frame.
        return pl.DataFrame(schema={
            "lhs": pl.String, "rhs": pl.String, "n_pairs": pl.Int64,
            "lhs_determines_rhs": pl.Boolean, "rhs_determines_lhs": pl.Boolean,
            "relationship": pl.String})
    return pl.DataFrame(recs)


def cardinality_hierarchy(frames):
    """Derive a grain hierarchy from the captured frames.

    Input: the bundle from Problem.to_frames() — "variables", "constraints",
    "constraint_rows", "grain_members", "entities".

    Output shape:
    column, row, row_type ("entity"|"entity_ref"|"grain"), node_id,
    parent_entity(nullable, a node_id), entity (display label),
    relationship_type(nullable), grain_id(nullable)

    An entity is top-level iff it's never the LHS of an n:1 (never
    functionally dependent on something else). n:n partners don't drag an
    entity down — cross-joining with a child doesn't make you one — but 1:1
    partners do (a 1:1 partner is the same identity, so whatever's true of
    it is true of you). Root entities that share an n:n/1:1 edge cluster
    into one column; unrelated root clusters get separate columns.

    Below each entity, grains are surfaced the moment every entity they
    span has been placed in that column (not just the entity that "closed"
    a strict parent chain) — a grain's span is often a proper subset of a
    node's full lineage, formed earlier in the pipeline before a later join
    happened, so this is a subset check against everything placed so far in
    the column, not an exact-match on the current path alone.

    A grain's entities aren't always literal ancestors on the branch it's
    surfaced on (e.g. it's placed under entity X, but also needs entity Y
    which actually lives elsewhere in the tree). Rather than hide that, a
    short "entity_ref" chain is inserted right above the grain for every
    such missing entity, in visit order -- a repeat of an entity that's
    placed for real elsewhere, existing only to make the grain's actual
    composition legible at the point it's shown. node_id disambiguates
    these from the real placement (f"{entity}__{grain_id}"); "entity" stays
    the plain display name in both places.
    """
    grain_members = frames["grain_members"]
    pairs = entity_pair_cardinality(frames)

    grain_entity_sets = sorted(
        (
            (row["grain_id"], set(row["entity"]))
            for row in grain_members.group_by("grain_id")
            .agg(pl.col("entity")).to_dicts()
        ),
        key=lambda g: len(g[1]),   # coarsest (fewest entities) first
    )

    entities = sorted(set(grain_members["entity"]))

    # child -> its n:1 parent. More than one n:1 target is a real ambiguity
    # (which one is "the" parent?) -- often a sign the data's too small or
    # degenerate for FD detection to mean anything (an entity with only one
    # distinct value is trivially "n:1" to everything). Break the tie toward
    # whichever candidate has more distinct values -- a single-value entity
    # carries no real partitioning information -- and say so out loud,
    # rather than silently picking or hard-failing the whole pipeline.
    n1 = pairs.filter(pl.col("relationship") == "n:1")
    parent_candidates = defaultdict(list)
    for lhs, rhs in zip(n1["lhs"], n1["rhs"]):
        parent_candidates[lhs].append(rhs)

    distinct_counts = dict(
        frames["constraint_rows"].group_by("key")
        .agg(pl.col("value").n_unique().alias("n")).iter_rows())
    parent_of = {}
    for e, candidates in parent_candidates.items():
        if len(candidates) > 1:
            chosen = max(candidates, key=lambda p: (distinct_counts.get(p, 0), p))
            print(f"[cardinality_hierarchy] ambiguous n:1 parent for {e!r}: "
                  f"{candidates} -> picked {chosen!r} (most distinct values)")
            parent_of[e] = chosen
        else:
            parent_of[e] = candidates[0]

    same_level = pairs.filter(pl.col("relationship").is_in(["1:1", "n:n"]))
    adjacency = defaultdict(set)
    for a, b in zip(same_level["lhs"], same_level["rhs"]):
        adjacency[a].add(b)
        adjacency[b].add(a)

    one_to_one = pairs.filter(pl.col("relationship") == "1:1")
    alias_adjacency = defaultdict(set)
    for a, b in zip(one_to_one["lhs"], one_to_one["rhs"]):
        alias_adjacency[a].add(b)
        alias_adjacency[b].add(a)

    def is_root(entity, seen):
        if entity in seen:
            return True
        seen.add(entity)
        if entity in parent_of:
            return False
        return all(is_root(alias, seen) for alias in alias_adjacency[entity])

    roots = [e for e in entities if is_root(e, set())]

    # cluster roots tied together by an n:n/1:1 edge into one column
    unclustered = set(roots)
    columns = []
    while unclustered:
        start = min(unclustered)
        stack, comp = [start], set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(p for p in adjacency[cur] if p in roots)
        unclustered -= comp
        columns.append(sorted(comp))
    columns.sort(key=lambda c: c[0])

    children_of = defaultdict(list)
    for child, parent in parent_of.items():
        children_of[parent].append(child)

    relationship_of = {(l, r): rel for l, r, rel in
                        zip(pairs["lhs"], pairs["rhs"], pairs["relationship"])}

    rows = []
    for col_idx, root_cluster in enumerate(columns):
        placed = set()
        placed_order = []   # same entities as `placed`, in visit order
        emitted_grains = set()
        row_idx = 0

        def place(entity, parent_id, relation, ancestors):
            nonlocal row_idx
            rows.append({
                "column": col_idx, "row": row_idx, "row_type": "entity",
                "parent_entity": parent_id, "entity": entity,
                "relationship_type": relation, "grain_id": None,
                "node_id": entity,
            })
            row_idx += 1
            placed.add(entity)
            placed_order.append(entity)
            here = ancestors + (entity,)

            for gid, gset in grain_entity_sets:
                if gid in emitted_grains:
                    continue
                if entity in gset and gset <= placed:
                    # label entities in the order the hierarchy visited them,
                    # not alphabetically -- reads as the flow down the tree
                    members = [e for e in placed_order if e in gset]

                    # a grain often needs entities that aren't literal
                    # ancestors on THIS branch (they're really placed
                    # elsewhere in the tree) -- show them as a short
                    # reference chain right above the grain rather than
                    # silently nesting the grain under an unrelated entity,
                    # so a reader doesn't have to hunt for what composes it
                    chain_parent = entity
                    for ref_entity in members:
                        if ref_entity in here:
                            continue
                        ref_id = f"{ref_entity}__{gid}"
                        rows.append({
                            "column": col_idx, "row": row_idx, "row_type": "entity_ref",
                            "parent_entity": chain_parent, "entity": ref_entity,
                            "relationship_type": "ref", "grain_id": None,
                            "node_id": ref_id,
                        })
                        row_idx += 1
                        chain_parent = ref_id

                    rows.append({
                        "column": col_idx, "row": row_idx, "row_type": "grain",
                        "parent_entity": chain_parent,
                        "entity": " x ".join(members),
                        "relationship_type": "grain", "grain_id": gid,
                        "node_id": gid,
                    })
                    row_idx += 1
                    emitted_grains.add(gid)

            for child in sorted(children_of.get(entity, [])):
                place(child, entity, relationship_of[(child, entity)], here)

        for e in root_cluster:
            place(e, None, "root", ())

    return pl.DataFrame(rows)


def model_graph(frames):
    """Render-agnostic IR: every entity/grain/constraint-family is a node,
    hierarchy placement and constraint reach are edges. No Mermaid, HTML, or
    any other rendering syntax lives here -- that's entirely the render
    layer's job, consuming {"nodes": DataFrame, "edges": DataFrame}. Swap
    the renderer (flowchart, erDiagram, plain text) without touching this.

    nodes: node_id, node_type ("entity"|"entity_ref"|"grain"|"constraint"), label,
      column (hierarchy column), row (hierarchy row, nullable on constraint
      nodes -- sort by (column, row) to recover the flat top-to-bottom
      traversal order, whether or not a renderer chooses to box columns
      apart), depth (nesting level, root=0, nullable on constraint nodes),
      parent_id (nullable), grain_id (nullable, the real grain_id -- only
      set on grain nodes), count (constraint instance count), example_ids
      (one raw var-id expr), example_fields (same expr with var ids
      swapped for entity names, so the *shape* of the constraint is
      legible without the row numbers).

    edges: source, target, edge_type ("hierarchy"|"touches"), label.
      Hierarchy edges carry the n:1/n:n/... relationship as their label.
      Touches edges are a constraint reaching a grain beyond its own
      application grain -- the cross-grain bridges. A constraint is always
      a node with 1..n touches edges out, never squeezed into a single
      binary line, so 3+ grain constraints need no special-casing.
    """
    hierarchy = cardinality_hierarchy(frames)
    cons = frames["constraints"]
    variables = frames["variables"]

    node_recs, edge_recs = [], []
    grain_column = {}
    depth_of = {}   # node_id -> nesting level, root = 0

    for row in hierarchy.iter_rows(named=True):
        node_id = row["node_id"]
        parent = row["parent_entity"]
        depth_of[node_id] = 0 if parent is None else depth_of[parent] + 1
        node_recs.append({
            "node_id": node_id, "node_type": row["row_type"], "label": row["entity"],
            "column": row["column"], "row": row["row"], "depth": depth_of[node_id],
            "parent_id": parent, "grain_id": row["grain_id"], "count": None,
            "example_ids": None, "example_fields": None,
        })
        if parent is not None:
            edge_recs.append({
                "source": parent, "target": node_id,
                "edge_type": "hierarchy", "label": row["relationship_type"],
            })
        if row["row_type"] == "grain":
            grain_column[row["grain_id"]] = row["column"]

    # ---- constraint families: one node per pipe-call site (type, grain,
    # call_id) -- NOT just (type, grain). Two different `.pipe(problem.add,
    # ...)` calls can share both a verb name and a home grain while doing
    # completely different things (a bound vs. a defining equality, say);
    # call_id is the only signal that actually tells them apart. ----
    var_to_entity = dict(zip(variables["var_id"], variables["entity"]))
    con_grains = constraint_grains(cons, variables)
    con_family = cons.select(["con_id", "type", "grain_id", "call_id"]).rename(
        {"grain_id": "app_grain"})
    family_grains = (
        con_grains.join(con_family, on="con_id", how="left")
        .group_by(["type", "app_grain", "call_id"])
        .agg(pl.col("grain_id").unique().alias("connects_grains")))
    family_meta = (
        cons.group_by(["type", "grain_id", "call_id"])
        .agg(pl.len().alias("count"), pl.col("expr").first().alias("example_ids"))
        .rename({"grain_id": "app_grain"}))
    # sort before assigning ids -- group_by order isn't guaranteed stable
    # across calls, and stable cf-ids matter for reproducible output/diffs
    families = (family_meta.join(family_grains, on=["type", "app_grain", "call_id"], how="left")
                .sort(["type", "app_grain", "call_id"]))

    for i, fam in enumerate(families.iter_rows(named=True)):
        cf_id = f"cf{i}"
        example_ids = fam["example_ids"] or ""
        example_fields = _VAR_RE.sub(
            lambda m: var_to_entity.get(m.group(0), m.group(0)), example_ids)
        node_recs.append({
            "node_id": cf_id, "node_type": "constraint",
            "label": f"{fam['type']} ×{fam['count']}",
            "column": grain_column.get(fam["app_grain"]), "row": None,
            "depth": depth_of.get(fam["app_grain"]),
            "parent_id": fam["app_grain"], "grain_id": None,
            "count": fam["count"], "example_ids": example_ids,
            "example_fields": example_fields,
        })
        for g in (fam["connects_grains"] or []):
            if g != fam["app_grain"]:
                edge_recs.append({
                    "source": cf_id, "target": g,
                    "edge_type": "touches", "label": None,
                })

    return {"nodes": pl.DataFrame(node_recs), "edges": pl.DataFrame(edge_recs)}
