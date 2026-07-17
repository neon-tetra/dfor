import polars as pl
from registry import Registry, VarStore, ConstraintStore


class Problem:
    def __init__(self, model):
        self._model = model
        self.registry = Registry()
        self.store = VarStore()
        self.constraints = ConstraintStore()

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
            grain = self._current_grain(df)
            ids = []
            for _ in range(df.height):
                id_ = self.store.next_id()
                var = model_attr(name=id_, **kwargs)
                self.store.put(id_, var, col_name, grain)
                ids.append(id_)
            df = df.with_columns(pl.Series(col_name, ids))
            self.registry.add(col_name, "satvar")
            return df
        return _piped

    # ---- constraint application ----

    def __init__(self, model):
        self._model = model
        self.registry = Registry()
        self.store = VarStore()
        self.constraints = ConstraintStore()
        self.diagnostic_mode = False
        self._assumption_lits = []

    def _make_constraint_verb(self, name, model_attr):
        def _piped(df, constraint_builder, **kwargs):
            self.observe(df)
            grain = self._current_grain(df)
            entities = self._satvar_cols(df)
            for row in df.iter_rows(named=True):
                row_keys = {k: v for k, v in row.items() if k in grain}
                rrow = self._resolve_row(row)
                args = constraint_builder(rrow)
                expr_str = ", ".join(str(a) for a in args)
                cname = self.constraints.next_name()

                ct = model_attr(*args)

                lit_index = None
                if self.diagnostic_mode:
                    lit = self._model.new_bool_var(cname)   # literal named after the constraint
                    try:
                        ct.only_enforce_if(lit)             # gate it
                        self._assumption_lits.append(lit)
                        lit_index = lit.index
                    except Exception:
                        pass                                # verb can't be gated -> ungated, still recorded

                self.constraints.put(cname, name, grain, entities, expr_str, row_keys, lit_index)
            return df
        return _piped

    def arm_diagnostics(self):
        """Call AFTER building the model, BEFORE solve, when diagnostic_mode was on."""
        if self._assumption_lits:
            self._model.add_assumptions(self._assumption_lits)

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
            groups[(rec["type"], rec["grain"], rec["entities"])].append(rec["row"])

        print(f"=== infeasibility core: {len(core)} constraints in {len(groups)} groups ===")
        for (ctype, grain, entities), rows in groups.items():
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
        sample = df[col].explode().drop_nulls()      # flatten list layer, drop nulls
        return len(sample) > 0 and self.store.is_id(sample[0])

    def _current_grain(self, df):
        return tuple(c for c in df.columns if not self._is_satvar_col(df, c))
    
    def _record_grain(self, df, col):
        sample = df[col].explode().drop_nulls()
        if len(sample) == 0 or not self.store.is_id(sample[0]):
            return
        id_ = sample[0]
        entity = self.store.entity_of(id_)
        born = self.store.birth_grain(id_)
        now = self._current_grain(df)
        folded = tuple(k for k in born if k not in now)
        self.registry.record_sighting(entity, born, now, folded)

    # ---- id -> var resolution before the user's lambda ----

    def _resolve_row(self, row):
        out = {}
        for col, val in row.items():
            out[col] = self._resolve_value(val)
        return out

    def _resolve_value(self, val):
        if isinstance(val, (list, pl.Series)):
            return [self._resolve_value(v) for v in val]
        return self.store.get(val) if self.store.is_id(val) else val
    
    def dump_grains(self):
        print("=== grain sightings ===")
        for entity, born, now, folded in self.registry.grains:
            print(f"{entity}: born {born} -> seen {now}  (folded: {folded})")
        print("=== entities ===")
        for name, e in self.registry._entities.items():
            print(f"{name}: {e.kind}")    