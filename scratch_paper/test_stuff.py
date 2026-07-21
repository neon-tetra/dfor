from ortools.sat.python import cp_model

# infeasible model with NAMED vars/constraints so we can see what survives
model = cp_model.CpModel()
xs = [model.new_int_var(0, 3, f"dforvar_{i}") for i in range(10)]
model.add(sum(xs) >= 25)     # infeasible vs below
model.add(sum(xs) <= 5)

print("=== ORIGINAL model ===")
print(f"  #vars={len(model.proto.variables)}  #constraints={len(model.proto.constraints)}")
for i, v in enumerate(model.proto.variables):
    print(f"    var[{i}] name={v.name!r} domain={list(v.domain)}")

solver = cp_model.CpSolver()
solver.parameters.stop_after_presolve = True
solver.parameters.cp_model_presolve = True
solver.parameters.log_search_progress = True
status = solver.solve(model)
print("\nstatus after stop_after_presolve:", solver.status_name(status))

# --- KEY QUESTION 1: did the model object itself get mutated to the presolved form? ---
print("\n=== model.proto AFTER stop_after_presolve (mutated in place?) ===")
print(f"  #vars={len(model.proto.variables)}  #constraints={len(model.proto.constraints)}")
for i, v in enumerate(model.proto.variables):
    print(f"    var[{i}] name={v.name!r} domain={list(v.domain)}")

# --- KEY QUESTION 2: is there ANY accessor for a separate presolved/mapping model? ---
print("\n=== hunting for presolved-model / mapping accessors ===")
for name in dir(solver):
    if any(k in name.lower() for k in ("presolv", "map", "model", "wrapper")):
        print("  solver:", name)
sw = getattr(solver, "_CpSolver__solve_wrapper", None)
if sw is not None:
    print("  solve_wrapper attrs:")
    for name in dir(sw):
        if not name.startswith("__"):
            print("     ", name)

# --- KEY QUESTION 3: does the response's tightened_variables give the presolved domains? ---
solver.parameters.fill_tightened_domains_in_response = True
status2 = solver.solve(model)
print("\n=== tightened_variables under stop_after_presolve ===")
for v in solver.response_proto.tightened_variables:
    print(f"    name={v.name!r} domain={list(v.domain)}")