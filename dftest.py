import polars as pl
import json

with open(r"C:\Users\andre\AppData\Roaming\Bay 12 Games\Dwarf Fortress\units_dump.json") as f:
    units = json.load(f)

rows = []
for u in units:
    row = {}

    # scalars
    for k, v in u.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[k] = v

    # physical attrs
    for attr, vals in u.get("physical_attrs", {}).items():
        for subk, subv in vals.items():
            row[f"phys_{attr}_{subk}"] = subv

    # mental attrs
    for attr, vals in u.get("mental_attrs", {}).items():
        for subk, subv in vals.items():
            row[f"ment_{attr}_{subk}"] = subv

    # skills — one column per skill token, value = rating
    for sk in u.get("skills", []):
        row[f"skill_{sk['token']}"] = sk["rating"]

    # traits — positional array, just unroll by index
    for i, v in enumerate(u.get("traits", [])):
        row[f"trait_{i}"] = v

    # beliefs — one column per token, value = strength
    for b in u.get("beliefs", []):
        row[f"belief_{b['token']}"] = b["strength"]

    # needs — one column per token, value = need_level (deity needs keyed by token+deity_id)
    for n in u.get("needs", []):
        key = f"need_{n['token']}" if n["deity_id"] == -1 else f"need_{n['token']}_{n['deity_id']}"
        row[key] = n["need_level"]

    # deity ids — one boolean column per deity id seen
    for did in u.get("deity_ids", []):
        row[f"deity_{did}"] = True

    rows.append(row)

df = pl.DataFrame(rows, infer_schema_length=len(rows))
print(df.shape)
print(df.columns)
#print to csv
df.write_csv("units_dump.csv")
df_print = (df#select all columns beginning with "deity"
    .select([pl.col(c) for c in df.columns if c.startswith("deity_")] + [pl.col("id"), pl.col("name"), pl.col("race")])
)
#get deities, pivot, count adherents
deity_summary = (df_print
    .unpivot(index=["id", "name", "race"], columns=[c for c in df_print.columns if c.startswith("deity_")], variable_name="deity_id", value_name="adherent")
    .groupby("deity_id")
    .agg(pl.count("id").alias("adherents"))
)
print(deity_summary)