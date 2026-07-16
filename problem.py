import polars as pl
from registry import Registry, VarStore
from tracked_frame import TrackedFrame


class Problem:
    def __init__(self, model):
        self._model = model
        self.registry = Registry()
        self.store = VarStore()

    def __getattr__(self, name):
        model_attr = getattr(self._model, name)

        if name.startswith("new"):
            return self._make_var_verb(name, model_attr)
        if name.startswith("add"):
            return self._make_constraint_verb(name, model_attr)

        # unknown: naive passthrough, keep tracking alive
        def _passthrough(df, *args, **kwargs):
            self.observe(df)
            return TrackedFrame(df, self)
        return _passthrough

    def _make_var_verb(self, name, model_attr):
        def _piped(df, col_name, **kwargs):
            self.observe(df)
            ids = []
            for _ in df.iter_rows(named=True):
                var = model_attr(**kwargs)          # make the var on the model
                ids.append(self.store.add(var))     # store -> int id
            df = df.with_columns(pl.Series(col_name, ids))
            self.registry.add(col_name, "satvar")   # column holds ids
            return TrackedFrame(df, self)
        return _piped

    def _make_constraint_verb(self, name, model_attr):
        def _piped(df, *args, **kwargs):
            self.observe(df)
            # stub: constraint creation + id->var resolution comes next
            return TrackedFrame(df, self)
        return _piped

    def observe(self, df):
        for col in df.columns:
            self.registry.add(col, "scalar")   # stub kind; create-if-not-exists