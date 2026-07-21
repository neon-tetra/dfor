"""
This problem arises from a colour printing firm which produces a variety of products from thin board, 
including cartons for human and animal food and magazine inserts. Food products, for example, are often 
marketed as a basic brand with several variations (typically flavours). Packaging for such variations 
usually has the same overall design, in particular the same size and shape, but differs in a small proportion 
of the text displayed and/or in colour. For instance, two variations of a cat food carton may differ only in 
that on one is printed ‘Chicken Flavour’ on a blue background whereas the other has ‘Rabbit Flavour’ printed on a 
green background. A typical order is for a variety of quantities of several design variations. Because each variation 
is identical in dimension, we know in advance exactly how many items can be printed on each mother sheet of board, whose 
dimensions are largely determined by the dimensions of the printing machinery. 

Each mother sheet is printed from a template, 
consisting of a thin aluminium sheet on which the design for several of the variations is etched. The problem is to decide, 
firstly, how many distinct templates to produce, and secondly, which variations, and how many copies of each, to include 
on each template. The following example is based on data from an order for cartons for different varieties of dry cat-food.

Variation	Order Quantity
Liver	250,000
Rabbit	255,000
Tuna	260,000
Chicken Twin	500,000
Pilchard Twin	500,000
Chicken	800,000
Pilchard	1,100,000
Total	3,665,000
Each design of carton is made from an identically sized and shaped piece of board. 
Nine cartons can be printed on each mother sheet, and several different designs can be printed at once, on the same mother sheet. 
(Hence, at least 407,223 sheets of card will be required to satisfy these order quantities.) 
Because in this example there are more slots in each template (9) than there are variations (7), 
it would be possible to fulfil the order using just one template. This creates an enormous amount of waste card, 
however. We can reduce the amount of waste by using more templates; with three templates, the amount of waste 
produced is negligible. The problem is therefore to produce template plans which will minimize the amount of waste 
produced, for 1 template, 2 templates,… and so on.

It is permissible to work in units of say 1000 cartons, so that the order quantities become 250, 255, etc.
"""

import polars as pl
from ortools.sat.python import cp_model
from problem import Problem

SLOTS_PER_SHEET = 9
TEMPLATE_LB = 4
TEMPLATE_UB = 4

model = cp_model.CpModel()
problem = Problem(model)

problem.diagnostic_mode = True

labels = pl.DataFrame({
    "label_id": [0, 1, 2, 3, 4, 5, 6],
    #"label_name": ["Liver", "Rabbit", "Tuna", "Chicken Twin", "Pilchard Twin", "Chicken", "Pilchard"],
    "order_quantity": [250, 255, 260, 500, 500, 800, 1100]})

subproblems = pl.DataFrame({"subproblem_id": list(range(1, TEMPLATE_UB + 1))})

templates = (subproblems
    .join(pl.DataFrame({"template_idx": list(range(1, TEMPLATE_UB + 1))}), how="cross")
    .filter(pl.col("template_idx") <= pl.col("subproblem_id"))
    #create new template id, unique id per template
    #use row index
    .with_row_index("template_id")
    .select(["subproblem_id", "template_id"])
    .pipe(problem.new_int_var, "template_produced_var", lb=0, ub=1100))

slots = pl.DataFrame({"slot_id": list(range(SLOTS_PER_SHEET))})

subproblems_x_labels = (
    subproblems
    .join(labels, how="cross")
    .select(["subproblem_id", "label_id", "order_quantity"])
    .pipe(problem.new_int_var, "subproblem_label_produced_val", lb=0, ub=1100)
    .pipe(problem.new_int_var, "subproblem_label_overproduction_val", lb=0, ub=1100)
    .pipe(problem.add_max_equality, lambda row: (row["subproblem_label_overproduction_val"], 
                                                 row["subproblem_label_produced_val"] - row["order_quantity"], 
                                                 0)))

(subproblems_x_labels
    .pipe(problem.add, lambda row: (row["subproblem_label_produced_val"] >= row["order_quantity"],)))

template = (templates
    .pipe(problem.new_int_var, "template_produced_var", lb=0, ub=1100))

templates_x_labels = (
    templates
    .join(labels, how="cross")
    .pipe(problem.new_int_var, "template_label_qty_var", lb=1, ub=9)
    .pipe(problem.new_int_var, "template_label_produced_val", lb=0, ub=1100)
    .pipe(problem.add_multiplication_equality, lambda row: (row["template_label_produced_val"],
                                                            row["template_label_qty_var"], row["template_produced_var"]))
    .select(["subproblem_id", "template_id", "label_id", "order_quantity", "template_label_qty_var", "template_label_produced_val"]))
    
(templates_x_labels
    .group_by("subproblem_id", "template_id")
    .agg(pl.col("template_label_qty_var").alias("template_label_qty_list"))
    .pipe(problem.add, lambda row: (sum(row["template_label_qty_list"]) == SLOTS_PER_SHEET,)))

subproblems_x_templates_x_labels = (
    subproblems_x_labels
    .join(templates_x_labels, on=["subproblem_id", "label_id"])
    .select(["subproblem_id", "template_id", "label_id", "order_quantity", "subproblem_label_produced_val", 
             "subproblem_label_overproduction_val", "template_label_qty_var", "template_label_produced_val"])
    .group_by("subproblem_id", "label_id", "order_quantity")
    .agg(pl.col("template_label_produced_val").alias("template_label_produced_list")
        ,pl.col("subproblem_label_produced_val").first())
    .pipe(problem.add, lambda row: (row["subproblem_label_produced_val"] == sum(row["template_label_produced_list"]),)))

problem.minimize(pl.sum(subproblems_x_labels["subproblem_label_overproduction_val"]))

problem.arm_diagnostics()
solver = cp_model.CpSolver()
status = problem.solve(solver)

import report
bundle = report.report(problem, solver, status)
import model_view
frames = problem.to_frames()
model_view.to_html(frames, "C:\\neon_tetra\\active\\dfor\\csplib\\002_families.html")
model_view.to_tree_html(frames, "C:\\neon_tetra\\active\\dfor\\csplib\\002_tree.html")
model_view.to_mermaid_html(frames, "C:\\neon_tetra\\active\\dfor\\csplib\\002_flowchart.html")
model_view.to_mermaid(frames, "C:\\neon_tetra\\active\\dfor\\csplib\\002_families.mmd")
model_view.to_mermaid(frames, "C:\\neon_tetra\\active\\dfor\\csplib\\002_families.mmd")
import model_analysis
analysis_df = model_analysis.entity_pair_cardinality(frames)
analysis_df.write_csv("C:\\neon_tetra\\active\\dfor\\csplib\\002_families_cardinality_analysis.csv")
