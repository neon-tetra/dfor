"""Given a set of patients distributed in a number of hospital zones and an available nursing staff, one must assign a 
nurse to each patient in such a way that the work is distributed evenly between nurses. Each patient is assigned an acuity level 
corresponding to the amount of care he requires; the workload of a nurse is defined as the sum of the acuities of the patients he 
cares for. A nurse can only work in one zone and there are retrictions both on the number of patients assigned to a nurse and on the 
corresponding workload. We balance the workloads by minimizing their standard deviation.

This problem can be decomposed in two phases: nurse staffing that assigns nurses to zones and nurse-patient assignment 
that then assigns patients to nurses."""

import polars as pl
from tracked_frame import TrackedFrame


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

zones  = pl.DataFrame({"zone_id":  list(range(num_zones))})
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

