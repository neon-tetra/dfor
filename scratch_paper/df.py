
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # dfor/, for `from problem import Problem` etc.

import polars as pl
pl.Config.set_ascii_tables(True)  # Windows console (cp1252) can't render polars' default Unicode table borders
from ortools.sat.python import cp_model
from problem import Problem

WORLD_X = 14
WORLD_Y = 11

model = cp_model.CpModel()
problem  = Problem(model)

#zip together two ranges to create a grid of coordinates
world_cells_list = [(x, y) for x in range(WORLD_X) for y in range(WORLD_Y)]
world_cells = pl.DataFrame(world_cells_list, schema=["x", "y"])

TARGET_ROOM_SHAPE = (2, 2)
ROOM_W, ROOM_H = TARGET_ROOM_SHAPE

grid = (world_cells
    .pipe(problem.new_bool_var, "room_anchor_var")
    .pipe(problem.new_bool_var, "cell_dug_var"))



(grid
    .filter((pl.col("x") + ROOM_W + 1 >= WORLD_X) |
            (pl.col("y") + ROOM_H + 1 >= WORLD_Y))
    .pipe(problem.add, lambda row: (row["room_anchor_var"] == 0,)))


#a room starting on an anchor point and extending TARGET_ROOM_SHAPE cells
#creates both the room and its surrounding walls
grid_rooms = (grid
    .lazy()
    .rename({"x": "x_lhs", "y": "y_lhs", "room_anchor_var": "room_anchor_var_lhs", "cell_dug_var": "cell_dug_var_lhs"})
    .join(grid.lazy().rename({"x": "x_rhs", "y": "y_rhs", "room_anchor_var": "room_anchor_var_rhs", "cell_dug_var": "cell_dug_var_rhs"}), how="cross")
    #anchor sits at the top-left of the WALL ring, not the interior -- full
    #footprint is (ROOM_W+2) x (ROOM_H+2): offsets 0 and ROOM_W+1/ROOM_H+1
    #are the wall sides, offsets 1..ROOM_W / 1..ROOM_H are the interior.
    .filter((pl.col("x_rhs").is_between(pl.col("x_lhs"), pl.col("x_lhs") + ROOM_W + 1)) &
            (pl.col("y_rhs").is_between(pl.col("y_lhs"), pl.col("y_lhs") + ROOM_H + 1)))
    .collect())

(grid_rooms
    #walls: either edge on x (offset 0 or ROOM_W+1) OR either edge on y --
    #all four sides of the ring, not just the near two.
    .filter((pl.col("x_rhs") == pl.col("x_lhs")) | (pl.col("x_rhs") == pl.col("x_lhs") + ROOM_W + 1) |
            (pl.col("y_rhs") == pl.col("y_lhs")) | (pl.col("y_rhs") == pl.col("y_lhs") + ROOM_H + 1))
    .pipe(problem.add_conditional, "add", lambda row: (row["cell_dug_var_rhs"] == 0,),
                                          lambda row: (row["room_anchor_var_lhs"])))

(grid_rooms
    #interior: strictly between both wall edges on both axes -- a true
    #ROOM_W x ROOM_H block, not ROOM_W+1 x ROOM_H+1.
    .filter((pl.col("x_rhs") > pl.col("x_lhs")) & (pl.col("x_rhs") < pl.col("x_lhs") + ROOM_W + 1) &
            (pl.col("y_rhs") > pl.col("y_lhs")) & (pl.col("y_rhs") < pl.col("y_lhs") + ROOM_H + 1))
    .pipe(problem.add_conditional, "add", lambda row: (row["cell_dug_var_rhs"] == 1,),
                                          lambda row: (row["room_anchor_var_lhs"])))

model.add_no_overlap_2
blocked_x = [5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9]
blocked_y = [6, 7, 6, 7, 8, 5, 6, 5, 6, 7, 6]
blocked_cells = pl.DataFrame({"x": blocked_x, "y": blocked_y})
(grid
    .join(blocked_cells, on=["x", "y"], how="inner")
    .pipe(problem.add, lambda row: (row["cell_dug_var"] == 0,)))
#maximize number of rooms
anchor_vars = [problem.store.get(v) for v in grid["room_anchor_var"]]
problem.maximize(sum(anchor_vars))

import model_view
frames = problem.to_frames()
model_view.to_tree_html(frames, "df.html")

solver = cp_model.CpSolver()
status = solver.Solve(model)

from report import solved
resolved = solved(grid, problem, solver)
resolved.write_csv("resolved.csv")
dug_map = resolved.select(["x", "y", "cell_dug_var"]).pivot(index="y", on="x", values="cell_dug_var").sort("y")

dug_map.write_csv("dug_map.csv")
