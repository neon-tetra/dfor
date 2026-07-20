import polars as pl
from ortools.sat.python import cp_model


def _has_solution(problem, solver):
    """Is there any assignment to read? True for OPTIMAL/FEASIBLE, and for
    UNKNOWN that found an incumbent before timing out."""
    if not problem.store._objs:
        return False
    try:
        solver.value(next(iter(problem.store._objs)))
        return True
    except Exception:
        return False


def report(problem, solver, status_obj):
    """One general report. Always returns the same five frames.
    Each is populated if its facts exist for this solve, else empty.
    The consumer decides what's relevant — no status branching here."""
    frames = problem.to_frames()         
    from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE, UNKNOWN, INFEASIBLE
    if status_obj == INFEASIBLE:
        status = "INFEASIBLE"
    elif status_obj == OPTIMAL:
        status = "OPTIMAL"
    elif status_obj == FEASIBLE:
        status = "FEASIBLE"
    elif status_obj == UNKNOWN:
        status = "UNKNOWN"
    else:
        status = f"UNEXPECTED({status_obj})"
    #status = solver.status

    # --- outcome: one-row summary, always present ---
    r = solver.response_proto
    has_sol = _has_solution(problem, solver)
    outcome = pl.DataFrame([{
        "status": status,
        "walltime": solver.wall_time,
        "conflicts": r.num_conflicts,
        "branches": r.num_branches,
        "restarts": r.num_restarts,
        "bool_propagations": r.num_binary_propagations,
        "int_propagations": r.num_integer_propagations,
        "objective": solver.objective_value if has_sol else None,
        "best_bound": solver.best_objective_bound if has_sol else None,
        "has_solution": has_sol,
    }])

    # --- solution: var -> value, empty if none ---
    if has_sol:
        vals = [{"var_id": v, "value": solver.value(obj)}
                for v, obj in problem.store._objs.items()]
        solution = frames["variables"].join(pl.DataFrame(vals), on="var_id", how="left")
    else:
        solution = frames["variables"].with_columns(
            pl.lit(None, dtype=pl.Int64).alias("value"))

    # --- core: conflicting constraints, empty if solve succeeded ---
    core_idx = list(solver.sufficient_assumptions_for_infeasibility())
    if core_idx:
        core = (frames["constraints"]
                .join(pl.DataFrame({"lit_index": core_idx}), on="lit_index", how="inner"))
    else:
        core = frames["constraints"].head(0)   # same schema, zero rows

    return {**frames, "outcome": outcome, "solution": solution, "core": core}

def solved(df, problem, solver):
    """Return a copy of df with every satvar column (ids) replaced by
    the solver's integer values. Quick one-off for post-solve inspection."""
    import polars as pl

    def resolve_cell(v):
        if isinstance(v, list):
            return [resolve_cell(x) for x in v]
        return solver.value(problem.store.get(v)) if problem.store.is_id(v) else v

    out = df
    for col in df.columns:
        if problem._is_satvar_col(df, col):
            out = out.with_columns(
                pl.col(col).map_elements(resolve_cell, return_dtype=pl.Int64).alias(col))
    return out