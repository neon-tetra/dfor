import polars as pl
from ortools.sat.python import cp_model

from dfor import Problem, model_view


def _items(problem):
    return (pl.DataFrame({"item": [0, 1, 2]})
            .pipe(problem.new_bool_var, "picked"))


def test_feasible_model_solves():
    problem = Problem(cp_model.CpModel())
    items = _items(problem)
    items.pipe(problem.add, lambda row: (row["picked"] >= 0,))
    (items.group_by(pl.lit(1).alias("one"))
          .agg(pl.col("picked").alias("picked_list"))
          .pipe(problem.add, lambda row: (sum(row["picked_list"]) == 2,)))
    solver = cp_model.CpSolver()
    status = problem.solve(solver)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_infeasible_model_gives_conflict_set():
    problem = Problem(cp_model.CpModel())
    problem.diagnostic_mode = True
    items = _items(problem)
    items.pipe(problem.add, lambda row: (row["picked"] == 1,), con_name="all_picked")
    items.pipe(problem.add, lambda row: (row["picked"] == 0,), con_name="none_picked")
    assert problem.solve() == cp_model.INFEASIBLE
    conflict = problem.quickxplain()
    assert len(conflict) > 0


def test_mermaid_contains_grain_and_entity_names():
    problem = Problem(cp_model.CpModel())
    nurses = pl.DataFrame({"nurse": [0, 1]})
    shifts = pl.DataFrame({"shift": [0, 1, 2]})
    assigned = nurses.join(shifts, how="cross").pipe(problem.new_bool_var, "assigned")
    (assigned.group_by("shift")
             .agg(pl.col("assigned").alias("assigned_list"))
             .pipe(problem.add_exactly_one, lambda row: (row["assigned_list"],),
                   con_name="one_nurse_per_shift"))
    text = model_view.to_mermaid(problem.to_frames())
    assert text.startswith("flowchart")
    for name in ("nurse", "shift"):
        assert name in text
