
"""
A number of cars are to be produced; they are not identical, because different options are available as variants on the basic model. 
The assembly line has different stations which install the various options 
(air-conditioning, sun-roof, etc.). These stations have been designed to handle at most a 
certain percentage of the cars passing along the assembly line. Furthermore, the cars requiring a 
certain option must not be bunched together, otherwise the station will not be able to cope. 
Consequently, the cars must be arranged in a sequence so that the capacity of each station is never exceeded. 
For instance, if a particular station can only cope with at most half of the cars passing along the line, 
the sequence must be built so that at most 1 car in any 2 requires that option. The problem has been shown to be NP-complete (Gent 1999).

The format of the data files is as follows:

First line: number of cars; number of options; number of classes.
Second line: for each option, the maximum number of cars with that option in a block.
Third line: for each option, the block size to which the maximum number refers.
Then for each class: index no.; no. of cars in this class; for each option, whether or not this class requires it (1 or 0).
This is the example given in (Dincbas et al., ECAI88):

Example input:
10 5 6
1 2 1 2 1
2 3 3 5 5
0 1 1 0 1 1 0 
1 1 0 0 0 1 0 
2 2 0 1 0 0 1 
3 2 0 1 0 1 0 
4 2 1 0 1 0 0 
5 2 1 1 0 0 0 
"""
file_name = "C:\\neon_tetra\\active\\dfor\\csplib\\car_sequencing\\ProblemDataSet200to400\\pb_200_01.txt"
#file_name = "C:\\neon_tetra\\active\\dfor\\csplib\\car_sequencing\\ProblemDataSet200to400\\test.txt"


#classes - len(classes), 1 column (id), nothing else
#options - len(options), 3 columns (id, max_cars_per_block, block_size)
#class_options - len(classes) * len(options), 3 columns (class_id, option_id, option_required)

with open(file_name, "r") as f:
    lines = f.readlines()
    num_cars, num_options, num_classes = map(int, lines[0].split())
    
    option_limits = list(map(int, lines[1].split()))
    block_sizes = list(map(int, lines[2].split()))
    options = [[i, option_limits[i], block_sizes[i]] for i in range(num_options)]

    classes = []
    class_options = []
    for i in range(num_classes):
        class_info = list(map(int, lines[i + 3].split()))
        class_id = class_info[0]
        num_cars_in_class = class_info[1]
        options_required = class_info[2:]
        classes.append((class_id, num_cars_in_class))
        for j in range(num_options):
            class_options.append((class_id, j, options_required[j]))


import polars as pl

#to pl dataframes
classes = pl.DataFrame(classes, schema=["class_id", "num_cars_in_class"])
options = pl.DataFrame({"option_id": list(range(num_options)),
                        "max_cars_per_block": option_limits,
                        "block_size": block_sizes})
class_options = pl.DataFrame(class_options, schema=["class_id", "option_id", "option_required"])


from ortools.sat.python import cp_model

from problem import Problem
problem = Problem(cp_model.CpModel())

problem.diagnostic_mode = False

num_cars = classes["num_cars_in_class"].sum()
positions = pl.DataFrame({"line_position": list(range(num_cars))})

positions_x_classes = (
    positions
    .join(classes, how="cross")
    .pipe(problem.new_bool_var, "class_at_position_var")
    .select(["line_position", "class_id", "num_cars_in_class", "class_at_position_var"]))

(positions_x_classes
 .group_by("line_position")
 .agg(pl.col("class_at_position_var").alias("class_at_position_list"))
 .pipe(problem.add_exactly_one, lambda row: (row["class_at_position_list"],)))

positions_x_options = (
    positions
    .join(options, how="cross")
    .pipe(problem.new_bool_var, "option_at_position_val")
    .select(["line_position", "option_id", "max_cars_per_block", "block_size", "option_at_position_val"]))

positions_x_classes_x_options = (
    positions_x_classes
    .join(class_options, on="class_id")
    .join(positions_x_options, on=["line_position", "option_id"])
    .select(["line_position", "class_id", "option_id", "option_required", "class_at_position_var", "option_at_position_val"]))

(positions_x_classes_x_options
 .pipe(problem.add_conditional, "add",
       lambda row: (row["option_at_position_val"] == row["option_required"],),
       lambda row: (row["class_at_position_var"])))
                                 
window_constraints = (
    positions_x_options
    .join(positions_x_options, on="option_id", how="left", suffix="_rhs")
    .filter((pl.col("line_position_rhs") <= pl.col("line_position")) &
            (pl.col("line_position_rhs") >= pl.col("line_position") - pl.col("block_size") + 1))
    .group_by(["option_id", "line_position"])
    .agg(pl.col("option_at_position_val_rhs").alias("window_member_list"),
        pl.col("max_cars_per_block").first(),)
    .pipe(problem.add, lambda row: (sum(row["window_member_list"]) <= row["max_cars_per_block"],)))

(positions_x_classes
 .group_by("class_id")
 .agg(pl.col("class_at_position_var").alias("class_at_position_list"),
      pl.col("num_cars_in_class").first())
    .pipe(problem.add, lambda row: (sum(row["class_at_position_list"]) == row["num_cars_in_class"],)))

problem.arm_diagnostics()
solver = cp_model.CpSolver()

solver.parameters.max_time_in_seconds = 60
solver.parameters.log_search_progress = True

solver, status = problem.solve(solver=solver)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solution found:")
    from report import solved
    solved_df = solved(positions_x_classes, problem, solver)
    result = (
        solved_df
        .filter(pl.col("class_at_position_var")==1)
        .sort("line_position")
        .select(["line_position", "class_id"])
    )
    print(result)
    #print_to_csv
    result.write_csv("C:\\neon_tetra\\active\\dfor\\csplib\\car_sequencing\\ProblemDataSet200to400\\pb_200_01_solution.csv")

frames = problem.to_frames()
import model_view
model_view.to_mermaid(frames, "C:\\neon_tetra\\active\\dfor\\csplib\\car_sequencing\\ProblemDataSet200to400\\pb_200_01.mmd")

for name, frame in bundle.items():
    print(f"\n=== {name} ({frame.height} rows) ===")
    print(frame)
