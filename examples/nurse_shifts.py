"""Nurse shift scheduling with dfor.

Wards have wings, wings have numbered rooms, and patients occupy rooms for stays of a few days. Every shift,
each working nurse takes one wing and a set of that wing's patients. Hard rules: coverage, one shift a
day, rest after nights, a rolling 7-day cap, contract min/max, patients and severity per nurse. Soft:
walking (the span of room numbers a nurse covers), continuity of care (same nurse for a patient on the
same shift the next day), and requested days off.

Data lives in nurse_data.py. Run from this folder: python nurse_shifts.py
"""

from pathlib import Path

import polars as pl
from ortools.sat.python import cp_model

from dfor import Problem
from dfor.report import solved
from dfor.model_view import to_tree_html
from dfor.matrix_view import to_matrix_html

from nurse_data import DAYS, SHIFTS, ROOMS, NURSE_WINGS, DAY_OFF_REQUESTS, PATIENT_STAYS

# ---- rules and weights ----
PATIENTS_MIN  = {"D": 3, "E": 3, "N": 4}             # per working nurse
PATIENTS_MAX  = {"D": 6, "E": 6, "N": 8}
SEVERITY_CAP  = {"D": 16, "E": 16, "N": 20}
WINDOW_DAYS, WINDOW_CAP = 7, 5                       # at most 5 shifts in any 7 days
CONTRACT_MIN, CONTRACT_MAX = 6, 10                   # shifts over the horizon
W_WALK, W_CONTINUITY, W_DAY_OFF = 1, 4, 20
TIME_LIMIT    = 60

model = cp_model.CpModel()
problem = Problem(model)


# =============================================================================
# Data
# =============================================================================
slots = pl.DataFrame([(d, k, d * len(SHIFTS) + i) for d in range(DAYS) for i, k in enumerate(SHIFTS)],
                     schema={"day": pl.Int64, "shift": pl.Utf8, "slot": pl.Int64}, orient="row")
nurse_wings = pl.DataFrame(NURSE_WINGS, schema={"nurse": pl.Int64, "wing": pl.Utf8}, orient="row")
day_off_requests = pl.DataFrame(DAY_OFF_REQUESTS, schema={"nurse": pl.Int64, "day": pl.Int64}, orient="row")
patients = pl.DataFrame(PATIENT_STAYS, schema={"patient": pl.Int64, "wing": pl.Utf8, "room": pl.Int64,
                                               "first_day": pl.Int64, "last_day": pl.Int64, "severity": pl.Int64},
                        orient="row")

patient_slots = (patients.join(slots, how="cross")
    .filter(pl.col("day").is_between(pl.col("first_day"), pl.col("last_day")))
    .select("patient", "wing", "room", "severity", "day", "shift", "slot"))


# =============================================================================
# Decisions
# =============================================================================
# works: nurse n takes wing w on slot s (only wings n is qualified for)
works = (nurse_wings.join(slots, how="cross")
    .pipe(problem.new_bool_var, "works"))
problem.label_grain(["nurse", "wing", "day", "shift", "slot"], "nurse_x_slot_x_wing")

# care: nurse n covers patient p on slot s (only nurses qualified for p's wing)
care = (patient_slots.join(nurse_wings, on="wing")
    .pipe(problem.new_bool_var, "care"))
problem.label_grain(["patient", "wing", "room", "severity", "day", "shift", "slot", "nurse"], "patient_x_slot_x_nurse")


# =============================================================================
# Hard rules
# =============================================================================
(care.group_by("patient", "slot").agg(pl.col("care").alias("list_care"))
    .pipe(problem.add_exactly_one,
          lambda r: (r["list_care"],),
          con_name="every_patient_covered_once"))

(care.join(works.select("nurse", "wing", "slot", "works"), on=["nurse", "wing", "slot"])
    .pipe(problem.add,
          lambda r: (r["care"] <= r["works"],),
          con_name="care_only_on_the_wing_you_work"))

(works.group_by("nurse", "day").agg(pl.col("works").alias("list_works"))
    .pipe(problem.add,
          lambda r: (sum(r["list_works"]) <= 1,),
          con_name="one_shift_a_day"))

# a night shift is followed by no day shift the next morning
nights = works.filter(pl.col("shift") == "N").group_by("nurse", "day").agg(pl.col("works").alias("list_night"))

mornings = (works.filter(pl.col("shift") == "D").with_columns(pl.col("day") - 1)
    .group_by("nurse", "day").agg(pl.col("works").alias("list_next_morning")))

(nights.join(mornings, on=["nurse", "day"])
    .pipe(problem.add,
          lambda r: (sum(r["list_night"]) + sum(r["list_next_morning"]) <= 1,),
          con_name="rest_after_nights"))

windows = pl.DataFrame({"window_start": list(range(DAYS - WINDOW_DAYS + 1))})

(works.join(windows, how="cross")
    .filter(pl.col("day").is_between(pl.col("window_start"), pl.col("window_start") + WINDOW_DAYS - 1))
    .group_by("nurse", "window_start").agg(pl.col("works").alias("list_works"))
    .pipe(problem.add,
          lambda r: (sum(r["list_works"]) <= WINDOW_CAP,),
          con_name="rolling_window_cap"))

(works.group_by("nurse").agg(pl.col("works").alias("list_works"))
    .pipe(problem.add,
          lambda r: (sum(r["list_works"]) >= CONTRACT_MIN,),
          con_name="contract_min")
    .pipe(problem.add,
          lambda r: (sum(r["list_works"]) <= CONTRACT_MAX,),
          con_name="contract_max"))

# patients and severity per working nurse
load = (care.group_by("nurse", "wing", "slot").agg(pl.col("care").alias("list_care"), pl.col("severity").alias("list_severity"))
    .join(works, on=["nurse", "wing", "slot"], how="right")
    .with_columns(min_patients=pl.col("shift").replace_strict(PATIENTS_MIN), max_patients=pl.col("shift").replace_strict(PATIENTS_MAX),
                  severity_cap=pl.col("shift").replace_strict(SEVERITY_CAP)))
(load
    .pipe(problem.add,
          lambda r: (sum(r["list_care"] or []) >= r["min_patients"] * r["works"],),
          con_name="min_patients_per_nurse")
    .pipe(problem.add,
          lambda r: (sum(r["list_care"] or []) <= r["max_patients"] * r["works"],),
          con_name="max_patients_per_nurse")
    .pipe(problem.add,
          lambda r: (sum(c * s for c, s in zip(r["list_care"] or [], r["list_severity"] or [])) <= r["severity_cap"],),
          con_name="severity_cap_per_nurse"))


# =============================================================================
# Walking: the span of room numbers each nurse covers on a shift
# =============================================================================
# lo <= room and hi >= room for every patient covered; span = hi - lo. minimizing walk pulls lo / hi onto
# the outermost rooms covered (and to equal values when nothing is covered)
walk = (works
    .pipe(problem.new_int_var, "lowest_room",  lb=1, ub=ROOMS)
    .pipe(problem.new_int_var, "highest_room", lb=1, ub=ROOMS)
    .pipe(problem.new_int_var, "walk",         lb=0, ub=ROOMS - 1)
    .pipe(problem.add,
          lambda r: (r["walk"] == r["highest_room"] - r["lowest_room"],),
          con_name="walk_is_room_span"))

(care.join(walk.select("nurse", "wing", "slot", "lowest_room", "highest_room"), on=["nurse", "wing", "slot"])
    .pipe(problem.add_conditional, verb="add",
          constraint_builder=lambda r: (r["lowest_room"] <= r["room"],),
          condition_builder =lambda r: r["care"],
          con_name="lowest_room_at_most_each_covered_room")
    .pipe(problem.add_conditional, verb="add",
          constraint_builder=lambda r: (r["highest_room"] >= r["room"],),
          condition_builder =lambda r: r["care"],
          con_name="highest_room_at_least_each_covered_room"))


# =============================================================================
# Continuity: the same nurse for a patient on the same shift the next day
# =============================================================================
# broken >= care tomorrow - care today, for every nurse: 1 unless tomorrow's nurse also had them today
next_day = care.select("patient", "nurse", "shift", pl.col("day") - 1, pl.col("care").alias("care_tomorrow"))

continuity = (care.select("patient", "nurse", "shift", "day", "care").join(next_day, on=["patient", "nurse", "shift", "day"])
    .group_by("patient", "shift", "day").agg(pl.col("care").alias("list_today"), pl.col("care_tomorrow").alias("list_tomorrow"))
    .pipe(problem.new_bool_var, "continuity_broken"))

(continuity.explode("list_today", "list_tomorrow")
    .pipe(problem.add,
          lambda r: (r["continuity_broken"] >= r["list_tomorrow"] - r["list_today"],),
          con_name="continuity_broken_if_new_nurse"))


# =============================================================================
# Objective
# =============================================================================
day_off_worked = works.join(day_off_requests, on=["nurse", "day"])
var = problem.store.get
problem.minimize(W_WALK * sum(var(v) for v in walk["walk"])
                 + W_CONTINUITY * sum(var(v) for v in continuity["continuity_broken"])
                 + W_DAY_OFF * sum(var(v) for v in day_off_worked["works"]))


# =============================================================================
# Model views (written before the solve, so they survive a lockup)
# =============================================================================
VIEW_DIR = Path(__file__).resolve().parent / "nurse_views"    # the matrix view writes one page per grain beside its index
VIEW_DIR.mkdir(exist_ok=True)
frames = problem.to_frames()
to_matrix_html(frames, str(VIEW_DIR / "matrix_view.html"))
to_tree_html(frames, str(VIEW_DIR / "tree_view.html"))
print(f"model views: {VIEW_DIR / 'matrix_view.html'}, {VIEW_DIR / 'tree_view.html'}")


# =============================================================================
# Solve and report
# =============================================================================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT
solver.parameters.num_workers = 8
status = problem.solve(solver)
print(f"status {solver.status_name(status)}  objective {solver.objective_value}  bound {solver.best_objective_bound}  "
      f"time {solver.wall_time:.1f}s  vars {len(model.proto.variables)}  constraints {len(model.proto.constraints)}")

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    r_walk = solved(walk.select("nurse", "wing", "slot", "works", "walk"), problem, solver).filter(pl.col("works") == 1)
    r_cont = solved(continuity.select("patient", "shift", "day", "continuity_broken"), problem, solver)
    r_off  = solved(day_off_worked.select("nurse", "day", "works"), problem, solver)
    print(f"nurse-shifts worked {r_walk.height}, total walk {r_walk['walk'].sum()} (mean {r_walk['walk'].mean():.1f}), "
          f"continuity broken {r_cont['continuity_broken'].sum()} of {r_cont.height}, day-off requests worked {r_off['works'].sum()} of {r_off.height}")
