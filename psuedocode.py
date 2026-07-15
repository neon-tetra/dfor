import polars as pl
from ortools.sat.python import cp_model

agents = pl.DataFrame(...)
schedules = pl.DataFrame(...)

from dfor import dfor
dfor = dfor()
dfor.register_df(agents, "agents")
dfor.register_df(schedules, "schedules")

model = cp_model.CpModel()
dfor.model = model


agents_x_schedules = (
    agents
    .join(schedules, how="inner", on="skill")
    .pipe(dfor.add_vars, 
          model.new_bool_var, 
          "assignment", var_name_cols=["agent_id", "schedule_id"])
    .group_by("agent_id", "skill", "day")
    .agg(pl.col("assignment").alias("assignments"))
    .pipe(dfor.add_constraints, 
          model.add_at_most_one,
          lambda row: (row["assignments"],),
          cst_name=["agent_id", "day"])    
)

result = dfor.solve()
dfor.generate_model_report("report.md")
