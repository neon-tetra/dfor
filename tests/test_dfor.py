import polars as pl
import pytest
from ortools.sat.python import cp_model

from dfor import Problem, Snapshot, model_view


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


# ---- snapshots: tiny models with a unique solution (qty = 2 * item), solved to optimality ----

def _build(items=(0, 1, 2, 3), label=True, keys=("item",), pinned=True):
    problem = Problem(cp_model.CpModel())
    if label:
        problem.label_grain(["item", "group"], "items", keys=list(keys))
    df = (pl.DataFrame({"item": list(items)})
            .with_columns(group=pl.col("item") % 2)
            .pipe(problem.new_int_var, "qty", lb=0, ub=10))
    if pinned:    # unpinned, only a fix can determine the values
        df.pipe(problem.add, lambda row: (row["qty"] == row["item"] * 2,))
    return problem


def _solved(problem):
    solver = cp_model.CpSolver()
    assert problem.solve(solver) == cp_model.OPTIMAL
    return solver


def _snapshot():
    problem = _build()
    return problem.snapshot(_solved(problem))


def test_snapshot_round_trip_and_hint(tmp_path, capsys):
    path = tmp_path / "items.dfsnap"
    _snapshot().save(path)
    snap = Snapshot.load(path)
    assert snap.grains["items"].keys == ("item",)
    assert snap.grains["items"].frame.sort("item")["qty"].to_list() == [0, 2, 4, 6]

    problem = _build(items=(1, 2, 3, 4))    # item 0 dropped, item 4 new
    problem.hint_grain("items", snap)
    assert "hint items.qty: 3 matched, 1 new, 1 dropped" in capsys.readouterr().out
    _solved(problem)


def test_fix_grain_pins_matched_rows():
    snap = _snapshot()
    problem = _build(pinned=False)
    problem.fix_grain("items", snap)
    solver = _solved(problem)
    assert [solver.value(problem.store.get(v)) for v in problem.store.to_frame()["var_id"]] == [0, 2, 4, 6]


def test_snapshot_rejects_duplicate_keys():
    problem = _build(keys=("group",))    # group has only two values across four items
    with pytest.raises(ValueError, match="share key values"):
        problem.snapshot(_solved(problem))


def test_snapshot_rejects_grain_labeled_after_its_variables():
    problem = _build(label=False)
    problem.label_grain(["item", "group"], "items")
    with pytest.raises(ValueError, match="label_grain"):
        problem.snapshot(_solved(problem))


def test_hint_rejects_key_mismatch():
    problem = _build(keys=("item", "group"))
    with pytest.raises(ValueError, match="snapshot keys"):
        problem.hint_grain("items", _snapshot())


def test_label_keys_must_be_subset_of_grain():
    with pytest.raises(ValueError, match="not a subset"):
        Problem(cp_model.CpModel()).label_grain(["item"], "items", keys=["other"])


def test_grain_labels_frame_includes_keys():
    frames = _build(keys=("item",)).to_frames()
    assert frames["grain_labels"].row(0, named=True)["keys"] == ["item"]
