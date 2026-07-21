"""
Proposed by Toby Walsh

An order m quasigroup is a Latin square of size m. That is, a m×m
 multiplication table in which each element occurs once in every row and column. For example,

1    2   3   4
4    1   2   3
3    4   1   2
2    3   4   1
is an order 4 quasigroup. A quasigroup can be specified by a set and a binary multiplication operator, * defined over this set. Quasigroup existence problems determine the existence or non-existence of quasigroups of a given size with additional properties. Certain existence problems are of sufficient interest that a naming scheme has been invented for them. We define two new relations, *321 and *312 by a∗321b=c
 iff c∗b=a
 and a∗312b=c
 iff b∗c=a
.

QG1.m problems are order m quasigroups for which if a∗b=c
, a∗b=c∗d
 and a∗321b=c∗321d
 then a=c
 and b=d
.

QG2.m problems are order m quasigroups for which if a*b=c*d and a *312 b = c *312 d then a=c and b=d.

QG3.m problems are order m quasigroups for which (a∗b)∗(b∗a)=a
.

QG4.m problems are order m quasigroups for which (b∗a)∗(a∗b)=a
.

QG5.m problems are order m quasigroups for which ((b∗a)∗b)∗b=a
.

QG6.m problems are order m quasigroups for which (a∗b)∗b=a∗(a∗b)
.

QG7.m problems are order m quasigroups for which (b∗a)∗b=a∗(b∗a)
.

For each of these problems, we may additionally demand that the quasigroup is idempotent. That is, a*a=a for every element a.

"""



import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # dfor/, for `from problem import Problem` etc.
_HERE = os.path.dirname(os.path.abspath(__file__))

from ortools.sat.python import cp_model
from problem import Problem
import polars as pl

model = cp_model.CpModel()
problem = Problem(model)
problem.diagnostic_mode = False

n = 8

series = [i for i in range(n)]

grid = pl.DataFrame({"rows": [i for i in range(n) for j in range(n)],
                     "cols": [j for i in range(n) for j in range(n)],})

grid = (grid
    .pipe(problem.new_int_var, "cell_var", lb=0, ub=n-1)
    .select(["rows", "cols", "cell_var"]))

(grid
 .pipe(problem.add_all_different, lambda row: (row["cell_var"], row["rows"]))
 .pipe(problem.add_all_different, lambda row: (row["cell_var"], row["cols"])))

grid_permutations = (grid
    .with_columns([pl.col("rows").alias("rows_lhs"),
                   pl.col("cols").alias("cols_lhs"),
                   pl.col("cell_var").alias("cell_var_lhs")])
    .join(grid, how="cross", suffix="_rhs")
    .filter((pl.col("rows_lhs") != pl.col("rows_rhs")) | (pl.col("cols_lhs") != pl.col("cols_rhs")))
    .sort(["rows_lhs", "cols_lhs", "rows_rhs", "cols_rhs"])
    .with_row_index("permutation_id")
    .with_columns([pl.col("cell_var_lhs").alias("a"),
                   pl.col("cell_var_rhs").alias("b")])
    .select(["permutation_id", "a", "b"])
    .pipe(problem.new_int_var, "lookup_index_a_b_val", lb=0, ub=(n*n)-1)
    .pipe(problem.new_int_var, "lookup_index_b_a_val", lb=0, ub=(n*n)-1)
    .pipe(problem.add, lambda row: (row["lookup_index_a_b_val"] == row["a"] * n + row["b"],))
    .pipe(problem.add, lambda row: (row["lookup_index_b_a_val"] == row["b"] * n + row["a"],))
    .pipe(problem.new_int_var, "a_b_val", lb=0, ub=n-1)
    .pipe(problem.new_int_var, "b_a_val", lb=0, ub=n-1)
    .sort("permutation_id")
    .with_columns(all_cell_values_list=pl.col("a").implode())
    .pipe(problem.add_element, lambda row: (row["lookup_index_a_b_val"], row["all_cell_values_list"], row["a_b_val"]))
    .pipe(problem.add_element, lambda row: (row["lookup_index_b_a_val"], row["all_cell_values_list"], row["b_a_val"]))
    .pipe(problem.new_int_var, "lookup_index_a_b_b_a_val", lb=0, ub=(n*n)-1)
    .pipe(problem.add, lambda row: (row["lookup_index_a_b_b_a_val"] == row["a_b_val"] * n + row["b_a_val"],))
    .pipe(problem.new_int_var, "a_b_b_a_val", lb=0, ub=n-1)
    .pipe(problem.add_element, lambda row: (row["lookup_index_a_b_b_a_val"], row["all_cell_values_list"], row["a_b_b_a_val"]))
    .pipe(problem.add, lambda row: (row["a_b_b_a_val"] == row["a"],)))
    


# --- diagnostic run: find the infeasibility, shrink it to a minimal core via
# QuickXplain, re-solve that core standalone, then dump everything (every
# frame + the tree view) -- overwriting whatever's already on disk from a
# previous run. ---
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 300.0
solver.parameters.log_search_progress = True
status = problem.solve(solver)
print("initial status:", solver.status_name(status))

import quickxplain

def _core_solver():
    s = cp_model.CpSolver()
    s.parameters.cp_model_presolve = False
    s.parameters.num_search_workers = 1
    return s

solver, status = quickxplain.reduce_to_core(problem, model, _core_solver)
print("reduced-core status:", solver.status_name(status))

frames = problem.to_frames()
for name, df in frames.items():
    # CSV can't hold the nested `entities` list column (constraints frame) --
    # join it into a plain string instead of dropping it.
    list_cols = [c for c in df.columns if isinstance(df.schema[c], pl.List)]
    if list_cols:
        df = df.with_columns([pl.col(c).list.join(", ").alias(c) for c in list_cols])
    out_path = os.path.join(_HERE, f"003_{name}.csv")
    df.write_csv(out_path)
    print(f"wrote {out_path}  ({df.height} rows)")

import model_view
tree_path = os.path.join(_HERE, "003_tree.html")
model_view.to_tree_html(frames, tree_path)
print(f"wrote {tree_path}")

#QG3.m problems are order m quasigroups for which (a∗b)∗(b∗a)=a