import polars as pl
from ortools.sat.python import cp_model

class Problem:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        # route to model / log stuff here later
        pass

    def make_name_from_cols(self, row, cols):
        if cols is None:
            return ""
        parts = [str(row[c]) if c in row else c for c in cols]
        return "_".join(parts)

    def add_vars(self, df: pl.DataFrame, fn, col_name: str, var_name_cols: list[str] | None = None, **kwargs) -> pl.DataFrame:
        def make_var(row_idx, row):
            name = self.make_name_from_cols(row, var_name_cols)
            name = f"{name}_{row_idx}"
            return fn(name=name, **kwargs)
        
        vars_ = [make_var(i, row) for i, row in enumerate(df.iter_rows(named=True))]
        return df.with_columns(pl.Series(col_name, vars_, dtype=pl.Object))

    def add_constraints(self, df: pl.DataFrame, fn, constraint_builder, cst_name: list[str] | None = None) -> pl.DataFrame:
        for row in df.iter_rows(named=True):
            name = self.make_name_from_cols(row, cst_name)
            fn(*constraint_builder(row)).with_name(name)
            
        return df

    def add_conditional_constraints(self, df, model, fn, constraint_builder, condition_builder, cst_name_cols: list[str] | None = None) -> pl.DataFrame:
        for row in df.iter_rows(named=True):
            name = self.make_name_from_cols(row, cst_name_cols)
            
            condition_expr = condition_builder(row)
            condition_var = model.new_bool_var(f"condition_{name}")
            model.add(condition_expr).only_enforce_if(condition_var).with_name(f"condition_{name}")

            principal_expr = constraint_builder(row)
            fn(principal_expr).only_enforce_if(condition_var).with_name(f"constraint_{name}")
            
        return df