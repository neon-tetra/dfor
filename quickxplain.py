"""Instrumented standalone group-level QuickXplain over dfor's assumption literals.
Prints progress on every oracle call so you can see whether it's stuck or just slow."""
import time
from collections import defaultdict
from ortools.sat.python import cp_model

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
        key = (rec["type"], rec["grain"], rec["entities"])
        groups[key].append(lit)
    return groups


def _short(key):
    """compact label for a group key so logs are readable."""
    ctype, grain, _ = key
    return f"{ctype}@{grain}"

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
        return

    _log("sanity passed. Beginning minimization.")
    minimal = _quickxplain(model, solver_factory, [], group_lists)

    survivor_keys = _keys_of(minimal)
    _log(f"=== MINIMAL CONFLICT: {len(survivor_keys)} of {len(keys)} groups ===")
    for k in survivor_keys:
        ctype, grain, entities = k
        print(f"  {ctype} @ {grain}  ×{len(groups[k])}")
        print(f"      entities={entities}", flush=True)