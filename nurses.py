"""Given a set of patients distributed in a number of hospital zones and an available nursing staff, one must assign a 
nurse to each patient in such a way that the work is distributed evenly between nurses. Each patient is assigned an acuity level 
corresponding to the amount of care he requires; the workload of a nurse is defined as the sum of the acuities of the patients he 
cares for. A nurse can only work in one zone and there are retrictions both on the number of patients assigned to a nurse and on the 
corresponding workload. We balance the workloads by minimizing their standard deviation.

This problem can be decomposed in two phases: nurse staffing that assigns nurses to zones and nurse-patient assignment 
that then assigns patients to nurses."""

import polars as pl
from tracked_frame import TrackedFrame

###helper functions to be tweaked and folded into dfor
def make_name_from_cols(row, cols):
    if cols is None:
        return ""
    parts = [str(row[c]) if c in row else c for c in cols]
    return "_".join(parts)

def add_vars(df: pl.DataFrame, fn, col_name: str, var_name_cols: list[str] | None = None, **kwargs) -> pl.DataFrame:
    def make_var(row_idx, row):
        name = make_name_from_cols(row, var_name_cols)
        name = f"{name}_{row_idx}"
        return fn(name=name, **kwargs)
    
    vars_ = [make_var(i, row) for i, row in enumerate(df.iter_rows(named=True))]
    return df.with_columns(pl.Series(col_name, vars_, dtype=pl.Object))

def add_constraints(df: pl.DataFrame, fn, constraint_builder, cst_name_cols: list[str] | None = None) -> pl.DataFrame:
    for row in df.iter_rows(named=True):
        name = make_name_from_cols(row, cst_name_cols)
        fn(*constraint_builder(row)).with_name(name)
        
    return df

def add_conditional_constraints(df, model, fn, constraint_builder, condition_builder, cst_name_cols: list[str] | None = None) -> pl.DataFrame:
    for row in df.iter_rows(named=True):
        name = make_name_from_cols(row, cst_name_cols)
        
        condition_expr = condition_builder(row)
        condition_var = model.new_bool_var(f"condition_{name}")
        model.add(condition_expr).only_enforce_if(condition_var).with_name(f"condition_{name}")

        principal_expr = constraint_builder(row)
        fn(principal_expr).only_enforce_if(condition_var).with_name(f"constraint_{name}")
        
    return df
###

with open("C:\\neon_tetra\\learning\\cpbench\\069\\20zones.txt", "r") as f:
    lines = f.readlines()
    num_zones, num_nurses = map(int, lines[0].split())
    min_patients, max_patients, max_acuity = map(int, lines[1].split())
    zones = []
    for i in range(num_zones):
        zone_patients = []
        for acuity in map(int, lines[i + 2].split()):
            zone_patients.append(acuity)
        zones.append(zone_patients)
mean_acuity = sum(sum(zone) for zone in zones) / num_nurses
mean_acuity = round(mean_acuity)
#print all  
print(f"Number of zones: {num_zones}")
print(f"Number of nurses: {num_nurses}")
print(f"Min patients per nurse: {min_patients}")
print(f"Max patients per nurse: {max_patients}")
print(f"Max acuity per nurse: {max_acuity}")

nurses = pl.DataFrame({"nurse_id": list(range(num_nurses))})

patient_acuities = []
patient_zones = []
for zone_id, zone_patients in enumerate(zones):
    for acuity in zone_patients:
        patient_acuities.append(acuity)
        patient_zones.append(zone_id)

patients = pl.DataFrame({"patient_id": list(range(len(patient_acuities))),
                            "patient_acuity": patient_acuities,
                            "zone_id": patient_zones})

from ortools.sat.python import cp_model
model = cp_model.CpModel()

nurse_x_zones = (
    nurses
    .join(patients.select(["zone_id"]).unique(), how="cross")
    .pipe(add_vars, model.new_bool_var, "nurse_assigned_zone", var_name_cols=["nurse_id", "zone_id"])
)
(nurse_x_zones
 .group_by("zone_id")
 .agg(pl.col("nurse_assigned_zone").alias("nurse_assigned_zone_list"))
 .pipe(add_constraints, model.add_exactly_one, lambda row: (row["nurse_assigned_zone_list"],), cst_name_cols=["zone_id"])
)

nurse_x_patients = (
    nurses
    .join(patients, how="cross")
    .select(["nurse_id", "patient_id", "patient_acuity", "zone_id"])
    .pipe(add_vars, model.new_bool_var, "nurse_assigned_patient",      var_name_cols=["nurse_id", "patient_id"])
    .pipe(add_vars, model.new_int_var,  "patient_acuity_contribution", var_name_cols=["nurse_id", "patient_id"], lb=0, ub=max_acuity)
    .pipe(add_constraints, model.add_multiplication_equality, 
          lambda row: (row["patient_acuity_contribution"], row["nurse_assigned_patient"], row["patient_acuity"]), 
          cst_name_cols=["nurse_id", "patient_id"])
)
(nurse_x_patients
 .group_by("patient_id")
 .agg(pl.col("nurse_assigned_patient").alias("nurse_assigned_patient_list"))
 .pipe(add_constraints, model.add_exactly_one, lambda row: (row["nurse_assigned_patient_list"],), cst_name_cols=["patient_id"])
)
(nurse_x_patients
 .join(nurse_x_zones, on=["nurse_id", "zone_id"])
 .pipe(add_constraints, model.add_implication, lambda row: (row["nurse_assigned_patient"], row["nurse_assigned_zone"]), cst_name_cols=["nurse_id", "patient_id"])
)
nurse_workloads = (
    nurse_x_patients
    .group_by("nurse_id")
    .agg(pl.col("patient_acuity_contribution").alias("patient_acuity_contribution_list"),
         pl.col("nurse_assigned_patient").alias("nurse_assigned_patient_list"))
    .pipe(add_vars, model.new_int_var, "nurse_workload", var_name_cols=["nurse_id"], lb=0, ub=max_acuity * max_patients)
    .pipe(add_constraints, model.add, lambda row: (sum(row["patient_acuity_contribution_list"]) == row["nurse_workload"],), cst_name_cols=["nurse_id"])
    .pipe(add_vars, model.new_int_var, "nurse_patient_count", var_name_cols=["nurse_id"], lb=0, ub=max_patients)
        .pipe(add_constraints, model.add, lambda row: (sum(row["nurse_assigned_patient_list"]) == row["nurse_patient_count"],), cst_name_cols=["nurse_id"])
    .pipe(add_constraints, model.add, lambda row: (row["nurse_workload"] <= max_acuity,), cst_name_cols=["nurse_id"])
    .pipe(add_constraints, model.add, lambda row: (row["nurse_patient_count"] <= max_patients,), cst_name_cols=["nurse_id"])
    .pipe(add_constraints, model.add, lambda row: (row["nurse_patient_count"] >= min_patients,), cst_name_cols=["nurse_id"])
    .pipe(add_vars, model.new_int_var, "nurse_workload_deviation", var_name_cols=["nurse_id"], lb=0, ub=max_acuity * max_patients)
    .pipe(add_constraints, model.add_abs_equality, lambda row: (row["nurse_workload_deviation"], row["nurse_workload"] - mean_acuity,), cst_name_cols=["nurse_id"])
)
model.minimize(sum(nurse_workloads["nurse_workload_deviation"]))
solver = cp_model.CpSolver()
result_status = solver.Solve(model)
if result_status == cp_model.OPTIMAL or result_status == cp_model.FEASIBLE:
    print(f"Solution found with objective value: {solver.ObjectiveValue()}")
    nurse_assignments = []
    for row in nurse_x_zones.iter_rows(named=True):
        if solver.BooleanValue(row["nurse_assigned_zone"]):
            nurse_assignments.append((row["nurse_id"], row["zone_id"]))
    print("Nurse assignments to zones:")
    for nurse_id, zone_id in nurse_assignments:
        print(f"Nurse {nurse_id} assigned to Zone {zone_id}")

    patient_assignments = []
    for row in nurse_x_patients.iter_rows(named=True):
        if solver.BooleanValue(row["nurse_assigned_patient"]):
            patient_assignments.append((row["nurse_id"], row["patient_id"]))
    print("Patient assignments to nurses:")
    for nurse_id, patient_id in patient_assignments:
        print(f"Patient {patient_id} assigned to Nurse {nurse_id}")
else:
    print(result_status)
    print("Infeasible, metrics below:")
    for row in nurse_workloads.iter_rows(named=True):
        print(f"Nurse {row['nurse_id']}: Workload = {solver.Value(row['nurse_workload'])}, Patient Count = {solver.Value(row['nurse_patient_count'])}, Workload Deviation = {solver.Value(row['nurse_workload_deviation'])}")