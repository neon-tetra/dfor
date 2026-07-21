"""Instrumented standalone group-level QuickXplain over dfor's assumption literals.
Prints progress on every oracle call so you can see whether it's stuck or just slow."""
import time
from collections import defaultdict
from ortools.sat.python import cp_model

from registry import ConstraintStore

_T0 = None
_CALL = 0

def _log(msg):
    dt = time.time() - _T0
    print(f"[{dt:7.2f}s] {msg}", flush=True)


def _group_literals(problem):
    groups = defaultdict(list)
    for lit in problem._assumption_lits:
        cname = problem.constraints.name_for_litindex(lit.index)
        if cname is None:
            continue
        rec = problem.constraints.get(cname)
        # call_id matters here same as it did for model_graph's families:
        # two different `.pipe()` sites can share (type, grain, entities)
        # while doing unrelated things, so without it they'd get treated
        # as one atomic group and never get isolated from each other.
        key = (rec["type"], rec["grain_id"], rec["entities"], rec["call_id"])
        groups[key].append(lit)
    return groups


def _short(key):
    """compact label for a group key so logs are readable."""
    ctype, grain, _entities, call_id = key
    return f"{ctype}@{grain}#{call_id}"

def _infeasible_with(model, solver_factory, lit_subset, label=""):
    global _CALL
    _CALL += 1
    call_id = _CALL
    lits = [l for grp in lit_subset for l in grp]
    _log(f"  oracle#{call_id} START  {label}  "
         f"({len(lit_subset)} groups, {len(lits)} lits) -> solving...")

    try:
        model.clear_assumptions()
        cleared = True
    except AttributeError:
        cleared = False
    model.add_assumptions(lits)

    t = time.time()
    solver = solver_factory()
    status = solver.solve(model)
    solve_s = time.time() - t
    infeas = status == cp_model.INFEASIBLE
    _log(f"  oracle#{call_id} DONE   {label}  status={solver.status_name(status)}  "
         f"infeasible={infeas}  ({solve_s:.2f}s, clear={'yes' if cleared else 'NO'})")
    return infeas

def _quickxplain(model, solver_factory, background, candidates, depth=0):
    ind = "  " * depth
    bg_lbl = f"[bg:{len(background)}]"
    _log(f"{ind}QX depth={depth}  candidates={len(candidates)} "
         f"{[_short(k) for k in _keys_of(candidates)]}  {bg_lbl}")

    if len(candidates) == 1:
        _log(f"{ind}  -> single candidate, ESSENTIAL: {_short(_keys_of(candidates)[0])}")
        return list(candidates)

    mid = len(candidates) // 2
    c1, c2 = candidates[:mid], candidates[mid:]

    if _infeasible_with(model, solver_factory, background + c1, f"{ind}bg+c1"):
        _log(f"{ind}  c1 alone suffices, dropping c2 ({len(c2)} groups)")
        return _quickxplain(model, solver_factory, background, c1, depth + 1)
    if _infeasible_with(model, solver_factory, background + c2, f"{ind}bg+c2"):
        _log(f"{ind}  c2 alone suffices, dropping c1 ({len(c1)} groups)")
        return _quickxplain(model, solver_factory, background, c2, depth + 1)

    _log(f"{ind}  neither half suffices -> both contribute, pairwise recurse")
    m1 = _quickxplain(model, solver_factory, background + c2, c1, depth + 1)
    m2 = _quickxplain(model, solver_factory, background + m1, c2, depth + 1)
    return m1 + m2


# --- helper so logs can show which groups a literal-list-bundle corresponds to ---
_KEY_INDEX = {}   # id(literal-list) -> key, filled in minimize()

def _keys_of(candidates):
    return [_KEY_INDEX[id(g)] for g in candidates]


def minimize(problem, model, solver_factory):
    """Reduce the full assumption set down to a minimal conflicting group set.
    Returns the flat list of surviving con_ids (empty if the sanity check
    fails), so callers can act on it programmatically -- not just read logs.
    """
    global _T0, _CALL, _KEY_INDEX
    _T0 = time.time()
    _CALL = 0

    groups = _group_literals(problem)
    keys = list(groups.keys())
    group_lists = [groups[k] for k in keys]
    _KEY_INDEX = {id(g): k for g, k in zip(group_lists, keys)}

    _log(f"grouped into {len(keys)} groups:")
    for k in keys:
        _log(f"    {_short(k)}  ×{len(groups[k])}")

    _log("SANITY: re-solving FULL assumption set (should be INFEASIBLE)...")
    if not _infeasible_with(model, solver_factory, group_lists, "FULL-SET"):
        _log("!!! Full set NOT infeasible under re-solve. "
             "assumptions plumbing is broken (likely append-not-replace). STOPPING.")
        return []

    _log("sanity passed. Beginning minimization.")
    minimal = _quickxplain(model, solver_factory, [], group_lists)

    survivor_keys = _keys_of(minimal)
    _log(f"=== MINIMAL CONFLICT: {len(survivor_keys)} of {len(keys)} groups ===")
    survivor_con_ids = []
    for k, lits in zip(survivor_keys, minimal):
        ctype, grain, entities, call_id = k
        print(f"  {ctype} @ {grain} #{call_id}  ×{len(lits)}")
        print(f"      entities={entities}", flush=True)
        for lit in lits:
            cname = problem.constraints.name_for_litindex(lit.index)
            if cname is not None:
                survivor_con_ids.append(cname)
    return survivor_con_ids


def reduce_to_core(problem, model, solver_factory):
    """Run QuickXplain to find the minimal conflicting constraint set, then
    rebuild `problem` around ONLY those constraints as a plain, unconditional
    model -- no reification, no assumptions armed -- and do one final solve.

    From that point on `problem`/the returned solver are indistinguishable
    from any other small, already-solved dfor problem: report.report(),
    model_view.to_tree_html(), model_analysis.cardinality_hierarchy() all
    just work unmodified, no special-casing for "this came from a core."

    Why a fresh CpModel rather than pruning model.proto in place: con_index
    (and every variable index) is a *position* in the proto's constraint/
    variable lists. Deleting entries in place shifts everything after the
    deletion, silently invalidating indices we still need. Building forward
    -- copy the full proto, clear constraints, append only the survivors in
    order -- sidesteps that entirely. Variables are NOT pruned: constraint
    protos reference them by index too, so dropping any would mean
    remapping every reference for no real benefit; a few now-irrelevant
    variables sitting unused in the reduced model is harmless.

    Note: problem.store (vars), problem.grains, and problem.registry are
    left untouched -- they still reflect the ORIGINAL full model. Frames
    built from them (grain_members, entities, variables) may list grains/
    entities no surviving constraint touches anymore; model_analysis
    already treats "no constraint_rows evidence" as relationship=None
    rather than guessing, so this shows up as a few inert extra nodes in
    the tree, not a break. Pruning that fully is future work if it matters.
    """
    survivor_con_ids = minimize(problem, model, solver_factory)
    if not survivor_con_ids:
        raise ValueError("QuickXplain found no minimal core (sanity check failed "
                          "or nothing was infeasible) -- nothing to reduce to.")

    old_store = problem.constraints
    survivor_rows = [old_store.get(cid) for cid in survivor_con_ids]
    # stable order by ORIGINAL con_index, not group-discovery order, so the
    # rebuilt model's numbering is deterministic across runs
    survivor_rows.sort(key=lambda r: r["con_index"])

    core_model = cp_model.CpModel()
    core_model.proto.copy_from(model.proto)
    core_model.proto.constraints.clear()

    new_store = ConstraintStore(problem.ids)
    for new_index, rec in enumerate(survivor_rows):
        orig_proto = model.proto.constraints[rec["con_index"]]
        new_c = core_model.proto.constraints.add()
        new_c.copy_from(orig_proto)
        new_c.enforcement_literal.clear()   # strip reification -- unconditional now
        new_store.put(
            rec["con_id"], rec["type"], rec["grain_id"], rec["entities"],
            rec["expr"], rec["row"], rec["call_id"], new_index,
            lit_index=None,
        )

    problem._model = core_model
    problem.constraints = new_store
    problem.diagnostic_mode = False
    problem._assumption_lits = []

    solver = solver_factory()
    status = solver.solve(core_model)
    return solver, status
