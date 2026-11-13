"""Assign patients to nurses with dfor's pipe verbs, solve, and write the
Mermaid diagram of the model. Small inline data, adapted from the nurse-zone
scratch model."""
import os

import polars as pl
from ortools.sat.python import cp_model

from dfor import Problem, model_view

nurses = pl.DataFrame({"nurse_id": [0, 1]})
patients = pl.DataFrame({"patient_id": [0, 1, 2, 3], "acuity": [3, 1, 2, 2]})
MAX_ACUITY_PER_NURSE = 5

model = cp_model.CpModel()
problem = Problem(model)

assign = (nurses
    .join(patients, how="cross")
    .pipe(problem.new_bool_var, "assigned"))

# every patient gets exactly one nurse
(assign
    .group_by("patient_id")
    .agg(pl.col("assigned").alias("assigned_list"))
    .pipe(problem.add_exactly_one, lambda row: (row["assigned_list"],)))

# no nurse carries more than the acuity cap
(assign
    .group_by("nurse_id")
    .agg(pl.col("assigned").alias("assigned_list"), pl.col("acuity").alias("acuity_list"))
    .pipe(problem.add, lambda row: (
        cp_model.LinearExpr.weighted_sum(row["assigned_list"], row["acuity_list"])
        <= MAX_ACUITY_PER_NURSE,)))

solver = cp_model.CpSolver()
status = problem.solve(solver)
print("status:", solver.status_name(status))

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    from dfor.report import solved
    print(solved(assign, problem, solver).filter(pl.col("assigned") == 1)
          .select(["nurse_id", "patient_id"]).sort("patient_id"))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nurse_assignment.mmd")
model_view.to_mermaid(problem.to_frames(), out)
print("wrote", out)
