"""Structural analysis over captured diagnostic frames: cardinality + FD
analysis of grains, building an IR of how the model's entities relate.
"""
from collections import defaultdict

import polars as pl


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

    return pl.DataFrame(recs)


def cardinality_hierarchy(frames):
    """Derive a grain hierarchy from the captured frames.

    Input: the bundle from Problem.to_frames() — "variables", "constraints",
    "constraint_rows", "grain_members", "entities".

    Output shape:
    column, row, row_type, parent_entity(nullable), entity,
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

    # child -> its unique n:1 parent. More than one n:1 target is a real
    # ambiguity (which one is "the" parent?) -- surface it, don't guess.
    n1 = pairs.filter(pl.col("relationship") == "n:1")
    parent_candidates = defaultdict(list)
    for lhs, rhs in zip(n1["lhs"], n1["rhs"]):
        parent_candidates[lhs].append(rhs)
    ambiguous = {e: ps for e, ps in parent_candidates.items() if len(ps) > 1}
    if ambiguous:
        raise ValueError(
            f"entities with more than one n:1 target (ambiguous parent): {ambiguous}")
    parent_of = {e: ps[0] for e, ps in parent_candidates.items()}

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

        def place(entity, parent_entity, relation):
            nonlocal row_idx
            rows.append({
                "column": col_idx, "row": row_idx, "row_type": "entity",
                "parent_entity": parent_entity, "entity": entity,
                "relationship_type": relation, "grain_id": None,
            })
            row_idx += 1
            placed.add(entity)
            placed_order.append(entity)

            for gid, gset in grain_entity_sets:
                if gid in emitted_grains:
                    continue
                if entity in gset and gset <= placed:
                    # label entities in the order the hierarchy visited them,
                    # not alphabetically -- reads as the flow down the tree
                    members = [e for e in placed_order if e in gset]
                    rows.append({
                        "column": col_idx, "row": row_idx, "row_type": "grain",
                        "parent_entity": entity,
                        "entity": " x ".join(members),
                        "relationship_type": "grain", "grain_id": gid,
                    })
                    row_idx += 1
                    emitted_grains.add(gid)

            for child in sorted(children_of.get(entity, [])):
                place(child, entity, relationship_of[(child, entity)])

        for e in root_cluster:
            place(e, None, "root")

    return pl.DataFrame(rows)
