import polars as pl
from ortools.sat.python import cp_model
from registry import Registry, VarStore, ConstraintStore, Grains
from ids import Ids


class Problem:
    def __init__(self, model):
        self._model = model
        self.ids = Ids()
        self.registry = Registry()
        self.grains = Grains(self.ids)
        self.store = VarStore(self.ids)
        self.constraints = ConstraintStore(self.ids)
        self.diagnostic_mode = False
        self._assumption_lits = []

    def __getattr__(self, name):
        model_attr = getattr(self._model, name)
        if name.startswith("new"):
            return self._make_var_verb(name, model_attr)
        if name.startswith("add"):
            return self._make_constraint_verb(name, model_attr)

        def _passthrough(df, *args, **kwargs):
            self.observe(df)
            return df
        return _passthrough

    # ---- var birth ----
    def _make_var_verb(self, name, model_attr):
        def _piped(df, col_name, **kwargs):
            self.observe(df)
            grain_id = self.grains.id_for(self._current_grain(df))
            ids = []
            for _ in range(df.height):
                id_ = self.store.next_id()
                var = model_attr(name=id_, **kwargs)
                self.store.put(id_, var, col_name, grain_id)
                ids.append(id_)
            df = df.with_columns(pl.Series(col_name, ids))
            self.registry.add(col_name, "satvar")
            return df
        return _piped

    # ---- constraint application ----
    def _make_constraint_verb(self, name, model_attr):
        def _piped(df, constraint_builder, **kwargs):
            self.observe(df)
            grain_cols = self._current_grain(df)
            grain_id = self.grains.id_for(grain_cols)
            entities = self._satvar_cols(df)
            call_id = self.constraints.next_call_id()
            for row in df.iter_rows(named=True):
                row_keys = {k: v for k, v in row.items() if k in grain_cols}
                rrow = self._resolve_row(row)
                self._apply_constraint(
                    model_attr, name, rrow, row_keys, grain_id, entities,
                    constraint_builder, call_id, enforce_if=None)
            return df
        return _piped

    def _apply_constraint(self, model_verb, verb_name, rrow, row_keys,
                        grain_id, entities, constraint_builder, call_id, enforce_if=None):
        """Shared body: build the constraint, optionally gate it on `enforce_if`,
        then apply diagnostic gating + capture. Used by both the plain and
        conditional verbs so there's one code path, not two.

        `call_id` identifies which *pipe call* produced this row -- one id
        shared by every row from a single `.pipe(problem.add, ...)` site,
        distinct across sites. Two constraints can share a verb name and a
        grain (e.g. two separate `.add()` calls landing at the same grain)
        while doing completely different things; call_id is the only signal
        that actually tells them apart, since verb name + grain alone can't.

        `ct.index` (captured as `con_index`) is the constraint's own position
        in `self._model.proto.constraints` -- distinct from `lit_index`,
        which is the *reification literal's variable* index used for
        assumptions. con_index is what you need to slice a constraint back
        out of the model proto directly (e.g. to build a small standalone
        sub-model from just an infeasibility core), independent of whether
        diagnostic mode ever reified it.
        """
        args = constraint_builder(rrow)
        expr_str = ", ".join(str(a) for a in args)
        cname = self.constraints.next_name()

        ct = model_verb(*args)
        con_index = ct.index
        if enforce_if is not None:
            ct.only_enforce_if(enforce_if)        # the conditional gate

        lit_index = None
        if self.diagnostic_mode:
            lit = self._model.new_bool_var(cname)
            try:
                ct.only_enforce_if(lit)
                self._assumption_lits.append(lit)
                lit_index = lit.index
            except Exception:
                pass

        self.constraints.put(cname, verb_name, grain_id, entities,
                            expr_str, row_keys, call_id, con_index, lit_index)

    def add_conditional(self, df, verb, constraint_builder, condition_builder):
        """Apply `verb`'s constraint, enforced only when `condition`
        returns a true literal for that row.

        `condition` returns a *literal* — an existing bool var, or its
        .Not(). Reification (binding a bool to an expression's truth) is done by
        the user with new_bool_var + two add_conditional calls.
        """
        self.observe(df)
        grain_cols = self._current_grain(df)
        grain_id = self.grains.id_for(grain_cols)
        entities = self._satvar_cols(df)
        model_verb = getattr(self._model, verb)
        call_id = self.constraints.next_call_id()

        for row in df.iter_rows(named=True):
            row_keys = {k: v for k, v in row.items() if k in grain_cols}
            rrow = self._resolve_row(row)
            condition = condition_builder(rrow)          # a resolved literal
            self._apply_constraint(
                model_verb, f"{verb}_conditional", rrow, row_keys, grain_id,
                entities, constraint_builder, call_id, enforce_if=condition)
        return df
    
    def minimize(self, expr):
        """Record the objective; apply it only in live mode.
        In diagnostic mode we suppress it (an objective forces the degraded
        'all assumptions' core path and turns the solve into optimization)."""
        self._objective = ("minimize", expr)
        if not self.diagnostic_mode:
            self._model.minimize(expr)

    def maximize(self, expr):
        self._objective = ("maximize", expr)
        if not self.diagnostic_mode:
            self._model.maximize(expr)
            
    def arm_diagnostics(self):
        if self._assumption_lits:
            self._model.add_assumptions(self._assumption_lits)

    def solve(self, solver=None):
        solver = solver or cp_model.CpSolver()
        if self.diagnostic_mode:
            self.arm_diagnostics()
            solver.parameters.cp_model_presolve = False
            solver.parameters.num_search_workers = 1
        # else: leave params to the caller / defaults, so you can set
        #       timeouts, workers, log_search_progress yourself before calling.
        status = solver.solve(self._model)
        return status
    
    def explain(self, solver):
        core = list(solver.sufficient_assumptions_for_infeasibility())
        if not core:
            print("No infeasibility core returned.")
            return
        from collections import defaultdict
        groups = defaultdict(list)
        for idx in core:
            cname = self.constraints.name_for_litindex(idx)
            if cname is None:
                continue
            rec = self.constraints.get(cname)
            key = (rec["type"], rec["grain_id"], rec["entities"])
            groups[key].append(rec["row"])

        print(f"=== infeasibility core: {len(core)} constraints in {len(groups)} groups ===")
        for (ctype, grain_id, entities), rows in groups.items():
            grain = self.grains.entities_of(grain_id)
            print(f"  {ctype} @ {grain}  ×{len(rows)}")
            print(f"      entities={entities}")
            print(f"      e.g. rows: {rows[:3]}{' ...' if len(rows) > 3 else ''}")

    def _satvar_cols(self, df):
        return tuple(c for c in df.columns if self._is_satvar_col(df, c))

    # ---- reading the frame at each pipe boundary ----
    def observe(self, df):
        for col in df.columns:
            if self._is_satvar_col(df, col):
                self.registry.add(col, "satvar")
                self._record_grain(df, col)
            else:
                self.registry.add(col, "scalar")

    def _is_satvar_col(self, df, col):
        dt = df.schema[col]
        base = dt.inner if isinstance(dt, pl.List) else dt
        if base != pl.String:
            return False
        sample = df[col].explode().drop_nulls()
        return len(sample) > 0 and self.store.is_id(sample[0])

    def _current_grain(self, df):
        return tuple(
            c for c in df.columns
            if not self._is_satvar_col(df, c)
            and not isinstance(df.schema[c], pl.List)   # list cols are data, not grain
        )

    def _record_grain(self, df, col):
        sample = df[col].explode().drop_nulls()
        if len(sample) == 0 or not self.store.is_id(sample[0]):
            return
        id_ = sample[0]
        entity = self.store.entity_of(id_)
        born_id = self.store.birth_grain(id_)
        born = self.grains.entities_of(born_id)
        now = self._current_grain(df)
        folded = tuple(k for k in born if k not in now)
        self.registry.record_sighting(entity, born, now, folded)

    # ---- id -> var resolution before the user's lambda (HOT PATH) ----
    def _resolve_row(self, row):
        return {col: self._resolve_value(val) for col, val in row.items()}

    def _resolve_value(self, val):
        if isinstance(val, (list, pl.Series)):
            return [self._resolve_value(v) for v in val]
        return self.store.get(val) if self.store.is_id(val) else val

    # ---- report-time frames (the 'dump the store' surface) ----
    def to_frames(self):
        """Crystallize all captured state into a small relational bundle.
        This is the substrate the reporting layer queries."""
        return {
            "variables":      self.store.to_frame(),
            "constraints":    self.constraints.to_frame(),
            "constraint_rows": self.constraints.rows_to_frame(),
            "grain_members":  self.grains.to_frame(),
            "entities":       self.registry.entities_to_frame(),
        }
    
    def dump_grains(self):
        print("=== grain sightings ===")
        for entity, born, now, folded in self.registry.grains:
            print(f"{entity}: born {born} -> seen {now}  (folded: {folded})")
        print("=== entities ===")
        for name, e in self.registry._entities.items():
            print(f"{name}: {e.kind}")