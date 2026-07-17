from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Ground truth: x must be BOTH >=5 and <=3. Impossible. We know exactly why.
x = model.new_int_var(0, 10, "x")

# gate each conflicting constraint behind a named literal
lit_hi = model.new_bool_var("dforcon_x_ge_5")
lit_lo = model.new_bool_var("dforcon_x_le_3")
lit_red = model.new_bool_var("dforcon_x_ge_0_redundant")   # a NON-conflicting one

model.add(x >= 5).only_enforce_if(lit_hi)
model.add(x <= 3).only_enforce_if(lit_lo)
model.add(x >= 0).only_enforce_if(lit_red)   # always satisfiable, should NOT be in core

model.add_assumptions([lit_hi, lit_lo, lit_red])

y = model.new_int_var(0, 10, "y")
z = model.new_int_var(0, 10, "z")

y_gate = model.new_bool_var("y_gate")
z_gate = model.new_bool_var("z_gate")

model.add_multiplication_equality(y, [x, z]).only_enforce_if(y_gate)
model.add_exactly_one([lit_hi, lit_lo, lit_red]).only_enforce_if(z_gate)

solver = cp_model.CpSolver()
solver.parameters.cp_model_presolve = False     # our 0.1 contract: presolve OFF
status = solver.solve(model)

print("status:", solver.status_name(status))

# --- now the experiment: which accessor exists, and what does it return? ---
print("\n--- trying accessors ---")

# candidate 1: method on solver (newer API)
try:
    core = solver.sufficient_assumptions_for_infeasibility()
    print("solver.sufficient_assumptions_for_infeasibility() ->", list(core))
except Exception as e:
    print("method form failed:", repr(e))

# candidate 2: field on the response proto (older API)
try:
    resp = solver.response_proto
    core = list(resp.sufficient_assumptions_for_infeasibility)
    print("response_proto.sufficient... ->", core)
except Exception as e:
    print("proto field form failed:", repr(e))

# candidate 3: older CamelCase response accessor
try:
    resp = solver.ResponseProto()
    core = list(resp.sufficient_assumptions_for_infeasibility)
    print("ResponseProto().sufficient... ->", core)
except Exception as e:
    print("CamelCase form failed:", repr(e))

# --- how do we map returned values back to OUR literals? ---
print("\n--- literal index reference ---")
for lit in (lit_hi, lit_lo, lit_red):
    print(f"{lit.name}: index={lit.index}")