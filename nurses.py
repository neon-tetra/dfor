"""Given a set of patients distributed in a number of hospital zones and an available nursing staff, one must assign a
nurse to each patient in such a way that the work is distributed evenly between nurses. Each patient is assigned an acuity level
corresponding to the amount of care he requires; the workload of a nurse is defined as the sum of the acuities of the patients he
cares for. A nurse can only work in one zone and there are retrictions both on the number of patients assigned to a nurse and on the
corresponding workload. We balance the workloads by minimizing their standard deviation.

This problem can be decomposed in two phases: nurse staffing that assigns nurses to zones and nurse-patient assignment
that then assigns patients to nurses."""

import polars as pl
from ortools.sat.python import cp_model
from problem import Problem

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

model = cp_model.CpModel()
problem = Problem(model)

nurse_x_zones = (
    nurses
    .join(patients.select(["zone_id"]).unique(), how="cross")
    .pipe(problem.new_bool_var, "nurse_assigned_zone"))

(nurse_x_zones
 .group_by("zone_id")
 .agg(pl.col("nurse_assigned_zone").alias("nurse_assigned_zone_list"))
 .pipe(problem.add_exactly_one, lambda row: (row["nurse_assigned_zone_list"],)))

nurse_x_patients = (
    nurses
    .join(patients, how="cross")
    .select(["nurse_id", "patient_id", "patient_acuity", "zone_id"])
    .pipe(problem.new_bool_var, "nurse_assigned_patient")
    .pipe(problem.new_int_var, "patient_acuity_contribution", lb=0, ub=max_acuity)
    .pipe(problem.add_multiplication_equality,
          lambda row: (row["patient_acuity_contribution"], row["nurse_assigned_patient"], row["patient_acuity"])))

(nurse_x_patients
 .group_by("patient_id")
 .agg(pl.col("nurse_assigned_patient").alias("nurse_assigned_patient_list"))
 .pipe(problem.add_exactly_one, lambda row: (row["nurse_assigned_patient_list"],)))

(nurse_x_patients
 .join(nurse_x_zones, on=["nurse_id", "zone_id"])
 .pipe(problem.add_implication, lambda row: (row["nurse_assigned_patient"], row["nurse_assigned_zone"])))

nurse_workloads = (
    nurse_x_patients
    .group_by("nurse_id")
    .agg(pl.col("patient_acuity_contribution").alias("patient_acuity_contribution_list"),
         pl.col("nurse_assigned_patient").alias("nurse_assigned_patient_list"))
    .pipe(problem.new_int_var, "nurse_workload", lb=0, ub=max_acuity * max_patients)
    .pipe(problem.add, lambda row: (sum(row["patient_acuity_contribution_list"]) == row["nurse_workload"],))
    .pipe(problem.new_int_var, "nurse_patient_count", lb=0, ub=max_patients)
    .pipe(problem.add, lambda row: (sum(row["nurse_assigned_patient_list"]) == row["nurse_patient_count"],))
    .pipe(problem.add, lambda row: (row["nurse_workload"] <= max_acuity,))
    .pipe(problem.add, lambda row: (row["nurse_patient_count"] <= max_patients,))
    .pipe(problem.add, lambda row: (row["nurse_patient_count"] >= min_patients,))
    .pipe(problem.new_int_var, "nurse_workload_deviation", lb=0, ub=max_acuity * max_patients)
    .pipe(problem.add_abs_equality, lambda row: (row["nurse_workload_deviation"], row["nurse_workload"] - mean_acuity,)))


model.minimize(sum(problem.store.get(x) for x in nurse_workloads["nurse_workload_deviation"]))

solver = cp_model.CpSolver()
status = solver.Solve(model)

def val(cell):
    return solver.Value(problem.store.get(cell))

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"objective: {solver.ObjectiveValue()}")
    for row in nurse_x_zones.iter_rows(named=True):
        if solver.BooleanValue(problem.store.get(row["nurse_assigned_zone"])):
            print(f"nurse {row['nurse_id']} -> zone {row['zone_id']}")
    for row in nurse_x_patients.iter_rows(named=True):
        if solver.BooleanValue(problem.store.get(row["nurse_assigned_patient"])):
            print(f"patient {row['patient_id']} -> nurse {row['nurse_id']}")
else:
    print(f"status: {status} (infeasible)")
    for row in nurse_workloads.iter_rows(named=True):
        print(f"nurse {row['nurse_id']}: workload={val(row['nurse_workload'])} "
              f"count={val(row['nurse_patient_count'])} dev={val(row['nurse_workload_deviation'])}")

problem.dump_grains()