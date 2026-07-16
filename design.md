# dfor — Informal Design Doc

Status: **early / exploratory.** Nothing here is locked. Sections marked ⚠️ are open
questions we have NOT resolved yet. First goal is a working v0 against a real
scheduling model, not a stable public API.

## What it is
`dfor` is a relational, dataframe-native authoring + diagnostics layer for CP-SAT
(OR-Tools) models. You author models by piping polars dataframes through `dfor`
verbs; `dfor` captures enough structure along the way to reconstruct a detailed,
grain-organized report of the model and diagnose failures (esp. infeasibility).

Core thesis: **the grain is the ontological unit.** Every variable and constraint
lives at exactly one grain (the key-column set of the frame it was born in). Entity
identity is read off column names. The model is a set of grains forming a lattice
(refinement partial order + shared-key + cross-join relationships); constraints are
couplings that live at a grain and reach across the lattice.

## Core objects

### `Dfor` (the class) — instance is a "problem" / context / scenario
- You create one and tie a CP-SAT model to it: `problem.model = model` (or via
  constructor — ⚠️ decide which).
- Owns: the model, the registry of observed grains/entities, constraint/var records,
  lineage records.
- CP-SAT verb access via `__getattr__`: `problem.add_at_most_one`,
  `problem.new_bool_var`, etc. are NOT defined on `Dfor`; unknown-attribute lookups
  are caught and routed to the underlying `self.model` method, returning a
  pipe-ready callable.
  - MUST guard: dunder names (`__x__`) re-raise `AttributeError` immediately.
  - MUST validate: if `self.model` lacks the verb, raise a clear error naming both
    dfor and the model.
  - Benefit: never lags CP-SAT (new verbs work for free); captures the verb NAME as
    data for the report.

### Usage contract (the minimal thing the user must do)
1. Make a `Dfor` problem, attach a CP-SAT model.
2. Use polars dataframes.
3. Build with native `.pipe(problem.<verb>, ...)` for var/constraint creation.
4. Everything else (structure, lineage, diagnostics) is captured by us.

Constraint/var calling convention: **Convention B** — the lambda returns the
ARGUMENTS to splat into the model verb. Uniform across all verbs, including the
trailing-comma single-arg case:
```python
.pipe(problem.add_at_most_one, lambda row: (row["assignments"],), cst_name=[...])
.pipe(problem.add,             lambda row: (row["a"] == row["b"] + row["c"],))
```
⚠️ Enforce/document the trailing-comma contract. ⚠️ Decide whether one row may ever
emit multiple constraints (if yes, lambda returns list-of-arg-tuples?).

### `TrackedFrame` (composition wrapper) — ⚠️ big TBD on mechanics
- After the first `dfor` verb, the user holds a `TrackedFrame` (composition: has-a
  polars frame, is-NOT-a subclass of it), not a raw `pl.DataFrame`.
- Wraps common polars methods (`select`, `join`, `group_by`, `agg`, `with_columns`,
  `filter`, ...) so each returns a `TrackedFrame`, keeping tracking alive down the
  chain.
- `__getattr__` fallthrough for un-wrapped methods: call the inner frame, decide
  re-wrap-or-passthrough by return type (frame → rewrap; scalar/Series/other →
  passthrough).
- Known costs (accepted for v0): must forward broad surface; `isinstance(x,
  pl.DataFrame)` is False for wrapper; **lazy frames** (`.lazy()`/`.collect()`) —
  ⚠️ UNRESOLVED what a wrapped LazyFrame does.

## What we capture, and how
- **Grains/entities:** first time we see a frame through a verb, record its schema
  (columns + dtypes) and grain (key columns). Entity identity = column name.
  - Assumption (load-bearing, document as contract): **same column name ⇒ same
    entity; distinct entities ⇒ distinct names.** Column-name unification silently
    merges false-friends and fails to merge true-synonyms.
  - ⚠️ Possible escape hatch: explicit alias declaration for known synonyms.
- **Lineage:** because tracking persists after first verb, joins/group-bys/selects in
  the tracked region are observable. The initial (pre-first-verb) join is NOT
  witnessed, but its RESULT grain is visible at first-sighting, so entity/grain
  structure is inferable from columns (grain-as-truth). ⚠️ Join *type/cardinality*
  (inner vs cross) and false-friend disambiguation are the only things that truly
  need witnessing — decided (for now) to treat as garnish/edge-case.
- **Vars & constraints:** captured at the verb boundary (we make/apply them, so we
  see them). Verb name captured via the string routed through `__getattr__`.

## Variable storage 
- Cpsat variables automatically keyed and hashed/retrieved when constraint methods called - implementation details tbd


## Report / output (the actual point) — future, but design toward it
tbd

## Naming / ergonomics
- Class `Dfor`; instance conventionally short (`problem`, `p`). ⚠️ Finalize.
- Keep CP-SAT model at `problem.model` so the two never fight for the word "model";
  happy path never types `problem.model.<verb>` (uses `problem.<verb>` via getattr).
- Distribution name on PyPI still TBD (`dfor` if free — verify page 404 + similarity
  guard; `satchel`/`sia` were taken/blocked).

## Known fragilities (accepted for v0, isolated to the convention layer)
- User must follow the pipe/verb pattern; deviations silently degrade capture.
- Column-name entity identity assumption (above).
- `TrackedFrame` surface breadth + isinstance leak + lazy.
- These live in the *permissive-convention* layer, which is cheap to harden later
  with guardrails — deliberately NOT pre-hardening.

## Open questions checklist
- [ ] Model attached via constructor or `.model =` assignment?
- [ ] Lazy-frame behavior for `TrackedFrame`.
- [ ] Alias hook for synonym columns?
- [ ] Diagnostic mode (assumption literals) opt-in design.