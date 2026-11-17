# dfor

Build OR-Tools CP-SAT models as Polars method chains. Because the whole model is built through dfor, it can also draw the model and narrow down infeasibility for you, with no extra code.

A side project, v0.1. It drives the scheduling model in [schedule-architect](https://github.com/neon-tetra/schedule-architect); the API may still change.

---

## Example

Each frame row is one decision, and constraints are written as ordinary CP-SAT expressions:

```python
import polars as pl
from ortools.sat.python import cp_model
from dfor import Problem

model = cp_model.CpModel()
problem = Problem(model)
# nurse_wings, slots, windows: ordinary Polars frames

# one bool per nurse x wing x shift the nurse is qualified for
works = (nurse_wings.join(slots, how="cross")
    .pipe(problem.new_bool_var, "works"))

# at most one shift a day
(works.group_by("nurse", "day").agg(pl.col("works").alias("list_works"))
    .pipe(problem.add,
          lambda r: (sum(r["list_works"]) <= 1,),
          con_name="one_shift_a_day"))

# at most 5 shifts in any 7 days
(works.join(windows, how="cross")
    .filter(pl.col("day").is_between(pl.col("window_start"), pl.col("window_start") + 6))
    .group_by("nurse", "window_start").agg(pl.col("works").alias("list_works"))
    .pipe(problem.add,
          lambda r: (sum(r["list_works"]) <= 5,),
          con_name="rolling_window_cap"))
```

The full example, [`examples/nurse_shifts.py`](examples/nurse_shifts.py), schedules 55 nurses across three wings for two weeks: coverage, rest after nights, rolling caps, patient load and severity limits, plus soft goals for walking distance, continuity of care, and requested days off.

```bash
pip install -e .
python examples/nurse_shifts.py
```

It writes the model views to `examples/nurse_views/`, then solves with a 60-second limit (`TIME_LIMIT` in the script). The model is large (about 72,000 variables and 230,000 constraints), so on a slow or few-core machine the solver may not find a first solution within 60 seconds; raise `TIME_LIMIT` if the status comes back `UNKNOWN`.

---

## Design

**Method chaining, in the style of dplyr or pandas.** dfor adds verbs on a `Problem` object, used inside `.pipe()`. Polars and CP-SAT are imported and used as normal; dfor never wraps a frame, so any Polars operation works mid-chain.

**Every CP-SAT method passes straight through.** `Problem` defines no constraint methods of its own. `__getattr__` forwards any name to the underlying `CpModel`: `new_*` methods create one variable per row, and `add_*` methods apply one constraint per row. Anything CP-SAT supports works immediately, with no per-method code in dfor.

**No expression DSL.** A constraint is a lambda that returns the same arguments you would pass to `CpModel`, written the same way. dfor resolves each row's columns to the real CP-SAT variables and applies the lambda across every row.

**Variables are held by reference.** Frames hold string ids, not variable objects; dfor keeps the objects in its own store. Early versions stored CP-SAT variables directly in the frames, until `group_by` on those columns produced undefined behavior. I reported it to Polars, who confirmed object columns weren't meant to work there and closed the gap. Ids are also what make the next point possible.

**The model knows its own structure.** Since every variable and constraint passes through a `.pipe()` call, `Problem` records each one with its grain (the key columns of the frame it was built on), its source call, and its name. That record powers everything below.

---

## Model views and diagnostics

**Model views.** `to_frames()` exports the recorded model as Polars frames, and the view functions render it: an interactive tree (`to_tree_html`), a per-grain matrix (`to_matrix_html`), and Mermaid diagrams (`to_mermaid`). They can be written before solving, so the structure is available even if a solve stalls.

**Infeasibility diagnosis.** Set `problem.diagnostic_mode = True` before building constraints, and every constraint is gated behind an assumption literal. On an infeasible solve, `problem.explain(solver)` gives a quick summary, and `problem.quickxplain()` narrows it to a minimal conflicting set. Conflicts are reported by constraint and grain, with example rows, rather than as raw variable indices.

**Solved values.** `report.solved(df, problem, solver)` takes any frame that still holds variable ids and returns it with each one replaced by its solved value.

---

## Status

Working and in use. Next to explore:
- More automated debugging and model-testing tools
- Possibly a MathOpt backend, for MIP solvers

More detail on the internals is in [`design.md`](design.md).

---

## Install

Not on PyPI yet:

```bash
pip install git+https://github.com/neon-tetra/dfor
```

Requires Python 3.10+, `ortools`, and `polars`.

---

## License

MIT. See [LICENSE](LICENSE).
