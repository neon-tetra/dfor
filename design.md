# dfor — Design Doc

Status: **working v0, in active use** (driving the `schedule-architect` CP-SAT
model). This doc describes what's actually built and how the pieces fit
together, not a spec for what's planned — see **Enhancements** at the bottom
for that.

## Purpose

`dfor` is a relational, dataframe-native authoring + diagnostics layer for
CP-SAT (OR-Tools) models. It has exactly two jobs:

1. **Helper functions for a method-chain model-building style.** You build a
   CP-SAT model by piping polars dataframes through `dfor` verbs
   (`.pipe(problem.new_bool_var, ...)`, `.pipe(problem.add, ...)`, ...)
   instead of writing loops that create variables and constraints by hand.
2. **Model documentation and debug automation, for free.** Because every
   var/constraint is created *through* dfor, dfor sees enough to reconstruct
   a full structural picture of the model (which entities exist, how they
   relate, which constraints touch which grains) and to power infeasibility
   diagnosis — without the user writing any separate reporting or debugging
   code.

Core thesis, unchanged from the original sketch of this doc: **the grain is
the ontological unit.** Every variable and constraint lives at exactly one
grain (the key-column set of the frame it was born in). Entity identity is
read off column names — same column name means same entity, by convention
and by design, not by declaration.

## Pillar 1: the method-chain building style

### `Problem` (`problem.py`)
You create one and hand it a CP-SAT model at construction time:
`problem = Problem(model)`. The model itself is stored as `self._model`
(private/underscored) — there's no public `problem.model` attribute; the
model is reached exclusively through the verb machinery below, so nothing
about "which name means the raw model vs. the dfor wrapper" is ever
ambiguous.

CP-SAT verb access is via `__getattr__` fallthrough: `problem.new_bool_var`,
`problem.add_at_most_one`, `problem.add_element`, etc. are not defined on
`Problem` — unknown-attribute lookups resolve the same-named method on
`self._model` and wrap it:
- Names starting with `new` become **var-birth verbs**: piped as
  `.pipe(problem.new_int_var, "col_name", lb=0, ub=9)`, they create one CP-SAT
  var per row, store it, and write the var's string id into a new column.
- Names starting with `add` become **constraint verbs**: piped as
  `.pipe(problem.add, lambda row: (row["a"] == row["b"],))`, they call the
  underlying model verb once per row with whatever the lambda returns.
- Anything else just observes the frame (for grain/entity tracking) and
  passes it through unchanged.

This means dfor never lags CP-SAT — any verb the installed OR-Tools version
supports works immediately, with no per-verb code in dfor itself.

**Convention:** the constraint-builder lambda returns the **arguments to
splat into the model verb**, always as a tuple (trailing comma for the
single-arg case): `lambda row: (row["a"] == row["b"] + row["c"],)`. Uniform
across every verb.

**`add_conditional(df, verb, constraint_builder, condition_builder)`** is the
reified/conditional form: `condition_builder` returns an existing literal (a
bool var or its `.Not()`), and the constraint is gated on it via
`only_enforce_if`. Reifying an *expression's* truth into a fresh bool is left
to the user (`new_bool_var` + two `add_conditional` calls), not automated.

There is **no wrapper class** around the polars frames — an earlier design
sketch (`TrackedFrame`, wrapping every polars method to keep tracking alive
down the chain) turned out to be unnecessary. Verbs take and return plain
`pl.DataFrame`s; `.pipe()` alone keeps the chain flowing, and ordinary
polars — including `.lazy()`/`.collect()` mid-chain — works exactly as it
would with no dfor involved at all, since nothing is wrapped to begin with.

### Capture machinery (`registry.py`, `ids.py`)
- **`Ids`**: one monotonic counter per namespace (`var_0`, `con_0`, `grain_0`,
  `call_0`, ...). The string prefix is load-bearing, not cosmetic — it's how
  `VarStore.is_id()` recognizes a satvar column at a glance, and it's what
  makes ids self-describing in any dump/log.
- **`VarStore`**: the hot-path arena. Var *objects* live in a plain
  `id -> Var` dict (`get`/`is_id` are just dict ops, called on every row of
  every constraint application); metadata (entity name, birth grain) is
  appended to a list in parallel and crystallized to a frame only on demand
  (`to_frame()`).
- **`Grains`**: interns grains by the frozenset of their entity (column)
  names — "have I seen this grain before" is an O(1) dict lookup, and
  declaration order is preserved for stable, legible grain ids.
- **`Registry`**: tracks every column ever seen (`"satvar"` vs `"scalar"`)
  and records grain "sightings" (an entity's birth grain vs. wherever it's
  currently observed, plus which key columns got folded away by a
  `group_by` in between) — the raw material for lineage.
- **`ConstraintStore`**: one row per constraint application, keyed by a
  `con_id`, tagged with a `call_id` shared by every row from the *same*
  `.pipe()` call site — the only signal that distinguishes two constraints
  which happen to share a verb name and a grain but come from different
  places in the code.

### `to_frames()`
Crystallizes everything captured into five plain polars frames —
`variables`, `constraints`, `constraint_rows` (long-format row keys, values
stringified so mixed-dtype grains don't choke polars' inference),
`grain_members`, `entities`. This bundle is the substrate every
documentation/diagnostic tool below reads from; nothing downstream touches
the live model or the capture stores directly.

## Pillar 2: model documentation and debug automation

### Structural analysis (`model_analysis.py`)
Turns the `to_frames()` bundle into a render-agnostic structural IR:
- `constraint_variables` / `constraint_grains`: recovers which vars, and
  therefore which grains, each constraint actually touches — parsed back out
  of the captured `expr` string rather than threaded through separately at
  capture time.
- `entity_pair_cardinality`: for every pair of entities that share a grain,
  **empirically** classifies their relationship (`1:1` / `n:1` / `1:n` /
  `n:n`) from the real co-occurring values seen in `constraint_rows` — not a
  schema guess. Pairs with no evidence come back `relationship=None`.
- `cardinality_hierarchy`: derives a parent/child display hierarchy from
  that cardinality data (an entity is top-level iff it's never functionally
  dependent on something else), clustering root entities that are tied
  together by a `1:1`/`n:n` edge into shared columns, and inserting short
  "entity_ref" breadcrumb chains wherever a grain needs an entity that isn't
  a literal ancestor on the branch it's shown on.
- `model_graph`: the actual render-agnostic IR (`{"nodes": DataFrame,
  "edges": DataFrame}`) — entities/grains from the hierarchy, plus one
  constraint-family node per **(type, grain, call_id)** pipe-call site (not
  just type+grain — two different `.pipe(problem.add, ...)` sites can share
  both while doing unrelated things), with "touches" edges fanning out to
  every grain a constraint family reaches beyond its own home grain.

### Rendering (`model_view.py`)
Pure render layers over `model_graph`'s IR — no structural logic lives here,
so swapping renderers never touches the analysis above:
- `to_tree_html`: the main interactive view — an indented outline (entities
  → grains → constraint cards), with SVG connector lines drawn client-side
  between a constraint and any grain it touches beyond its own home grain.
- `to_tree_json`: the same tree, as plain JSON instead of HTML+SVG. Meant
  for grepping/scanning (by a human or an LLM) and — importantly — for
  dumping **before** a solve even starts, so the model's structure survives
  a search lockup or crash.
- `to_mermaid` / `to_mermaid_html`: a Mermaid flowchart rendering of the same
  IR, entities/grains nested in per-column subgraphs.
- `to_json` / `to_html` (`family_view`): an earlier, coarser grain-level card
  view (nodes = grains, edges = constraint families). Still present, mostly
  superseded by the tree renderers for day-to-day use.

### Solve results (`report.py`)
- `report(problem, solver, status_obj)`: always returns the same five-frame
  shape (`outcome`, `solution`, `core`, plus the raw `to_frames()` bundle),
  populated if the relevant facts exist for this solve and empty otherwise —
  deliberately no status branching pushed onto the caller.
- `solved(df, problem, solver)`: the quick one-off — hand it *any* frame
  still holding var-id-string columns and get back a copy with every satvar
  column resolved to its solved integer value (list-columns included).

### Infeasibility diagnosis
- `problem.diagnostic_mode = True`, set **before** building constraints,
  reifies every constraint behind a fresh assumption literal as it's
  created; `arm_diagnostics()` (called automatically inside `solve()`) wires
  those literals in, and forces `cp_model_presolve = False` /
  `num_search_workers = 1` — both required for OR-Tools to return a valid
  infeasibility core at all. `minimize()`/`maximize()` are suppressed while
  armed (an objective would force the degraded "all assumptions" path).
- `problem.explain(solver)`: a quick, un-minimized summary — groups the
  literals `sufficient_assumptions_for_infeasibility()` returns by
  (type, grain, entities) and prints example offending rows. Good first
  look; not guaranteed minimal.
- `quickxplain.py`: the heavier tool, for when `explain()` isn't precise
  enough. `minimize()` runs an instrumented, progress-logged QuickXplain
  over the *grouped* assumption literals (grouping by `call_id` too, same
  reasoning as constraint families above) to find a genuinely minimal
  conflicting set. `reduce_to_core()` goes one step further: rebuilds a
  fresh, ordinary (non-reified) `CpModel` containing only the surviving
  constraints and re-solves it — from that point on the reduced problem is
  an indistinguishable, normal small dfor problem, so `report.report()`,
  `model_view.to_tree_html()`, `model_analysis.cardinality_hierarchy()` all
  work on it completely unmodified, no special-casing for "this came from a
  core."

## Layout

| File | Role |
|---|---|
| `problem.py` | `Problem` — pillar 1's home, plus `to_frames()` |
| `registry.py` | Capture stores: `VarStore`, `Grains`, `Registry`, `ConstraintStore` |
| `ids.py` | Namespaced monotonic id generator |
| `model_analysis.py` | Structural IR: cardinality, hierarchy, render-agnostic graph |
| `model_view.py` | Render layers over that IR (tree/mermaid/card views) |
| `report.py` | Solve-result frames + the `solved()` resolver |
| `quickxplain.py` | Standalone infeasibility-core minimizer |
| `scratch_paper/` | Throwaway experiments; not part of the library surface |

No `__init__.py` exports and no packaging — consuming projects add this
directory straight to `sys.path` (a `.pth` file, see e.g.
`schedule-architect/CLAUDE.md`) and import modules by bare name
(`from problem import Problem`, `import model_view`), not as a package.
PyPI distribution was floated early on and never pursued; not currently
blocking anything.

## Known fragilities (accepted, not hardened)
- The user must actually follow the pipe/verb pattern — a var or constraint
  created any other way (calling `model.new_bool_var(...)` directly, say)
  is invisible to dfor and silently missing from every report.
- Column-name = entity identity is load-bearing and unchecked: two different
  real-world things sharing a column name get silently merged into one
  entity; the same thing under two different names gets silently split into
  two. No alias/synonym declaration exists yet (see Enhancements).
- These live in the *permissive-convention* layer, deliberately not
  pre-hardened — cheap to add guardrails to later if a real project trips
  over them.

## Enhancements (future, not yet built)

Ideas that fall directly out of what's already being captured, not
speculative feature requests:

- **Alias hook for synonym columns** — a way to declare "this column and
  that column are the same entity," closing the gap in the fragility above.
- **Structural diff between two solves or two model versions** — since
  `to_frames()` is already a stable relational bundle, `diff(frames_a,
  frames_b)` comparing grain/constraint counts and (via `solved()`) variable
  values would directly answer "did my last edit change what I think it
  changed," which came up repeatedly in the schedule-architect work as
  something done by hand via `benchmark_log.csv` and eyeballing.
- **Symmetry/redundancy hints from `entity_pair_cardinality`** — it already
  detects `1:1` relationships empirically; surfacing "these two entities are
  always 1:1 — modeled as the same grain?" as a lint-style note could catch
  missing symmetry-breaking opportunities before they cost solve time.
- **First-class hint scaffolding on `Problem`** — schedule-architect's
  `solver/hints.py` hand-rolled the "resolve a var-id column, call
  `model.add_hint`" pattern; once that pattern is proven out further, a
  small `problem.hint(df, col, value)` verb would fold it into the same
  capture/convention machinery as everything else.
- **Capture-time cost warnings** — schedule-architect spent real effort
  empirically discovering that `add_element`'s cost scales with array
  length, not call count. dfor already sees the verb name and args at the
  point of capture, so it could proactively flag "this `add_element` call's
  array argument has N entries" past some threshold, turning that hard-won
  lesson into an automatic check for future models.
- **Search/filter in `to_tree_html`** — the tree view is data-complete but
  static; a client-side filter (everything needed is already in the page)
  would help once a model's tree gets large.
