import io
import json
import zipfile
from dataclasses import dataclass
from functools import reduce

import polars as pl

FORMAT = "dfsnap"
VERSION = 1


@dataclass(frozen=True)
class Grain:
    keys: tuple          # columns that identify a row
    frame: pl.DataFrame  # all grain columns + one solved-value column per entity


@dataclass(frozen=True)
class Snapshot:
    grains: dict         # grain label -> Grain

    def save(self, path):
        manifest = {"format": FORMAT, "version": VERSION, "grains": {}}
        with zipfile.ZipFile(path, "w") as z:
            for i, (label, g) in enumerate(self.grains.items()):
                buf = io.BytesIO()
                g.frame.write_parquet(buf)
                z.writestr(f"{i}.parquet", buf.getvalue())
                manifest["grains"][label] = {"keys": list(g.keys), "file": f"{i}.parquet"}
            z.writestr("manifest.json", json.dumps(manifest))

    @classmethod
    def load(cls, path):
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json"))
            if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
                raise ValueError(f"{path} is not a {FORMAT} v{VERSION} snapshot")
            return cls({
                label: Grain(tuple(m["keys"]), pl.read_parquet(io.BytesIO(z.read(m["file"]))))
                for label, m in manifest["grains"].items()})


def take(problem, solver, labels=None):
    if labels is None:    # every labeled grain that has variables
        born = dict.fromkeys(m["birth_grain_id"] for m in problem.store._meta)
        labels = [problem.grains.label_of(g) for g in born
                  if problem.grains.is_labeled(problem.grains.entities_of(g))]
    return Snapshot({label: _grain(problem, solver, label) for label in labels})


def _retained(problem, label):
    gid = problem.grains.by_label(label)
    births = problem._births.get(gid, {})
    built = {m["entity"] for m in problem.store._meta if m["birth_grain_id"] == gid}
    if built - births.keys() or not births:
        raise ValueError(
            f"grain {label!r}: no data was kept for {sorted(built - births.keys()) or 'its variables'}, "
            f"because they were built before the grain was labeled -- "
            f"move the label_grain call above the new_* calls")
    return gid, births


def _grain(problem, solver, label):
    gid, births = _retained(problem, label)
    frames = [f.with_columns(pl.Series(col, [solver.value(problem.store.get(i)) for i in f[col]]))
              for col, f in births.items()]
    cols = list(problem.grains.entities_of(gid))
    frame = reduce(lambda a, b: a.join(b, on=cols, how="full", coalesce=True, nulls_equal=True), frames)
    keys = list(problem.grains.keys_of(gid))
    dups = frame.filter(frame.select(keys).is_duplicated())
    if dups.height:
        raise ValueError(
            f"grain {label!r}: {dups.height} rows share key values (keys {keys}), "
            f"e.g. {dups.select(keys).head(3).rows()} -- keys must identify a row; "
            f"add a column to the keys or remove the duplicate rows")
    return Grain(tuple(keys), frame)


def _matched(problem, label, snap, verb):
    """Per entity: the live birth frame (var ids) with the snapshot's value in
    `_value`, for rows whose keys match. Prints matched / new / dropped counts."""
    gid, births = _retained(problem, label)
    keys = list(problem.grains.keys_of(gid))
    g = snap.grains[label]
    if set(keys) != set(g.keys):
        raise ValueError(f"{label}: snapshot keys {list(g.keys)} != model keys {keys}")
    for col, live in births.items():
        if col not in g.frame.columns:
            continue
        saved = g.frame.select(*keys, pl.col(col).alias("_value"))
        joined = live.join(saved, on=keys, how="left", nulls_equal=True)
        matched = joined.filter(pl.col("_value").is_not_null())
        dropped = saved.join(live, on=keys, how="anti", nulls_equal=True).height
        print(f"{verb} {label}.{col}: {matched.height} matched, "
              f"{live.height - matched.height} new, {dropped} dropped")
        yield col, matched


def hint(problem, label, snap):
    for col, m in _matched(problem, label, snap, "hint"):
        for id_, v in zip(m[col], m["_value"]):
            problem._model.add_hint(problem.store.get(id_), v)


def fix(problem, label, snap):
    gid = problem.grains.by_label(label)
    keys = list(problem.grains.keys_of(gid))
    call_id = problem.constraints.next_call_id()
    for col, m in _matched(problem, label, snap, "fix"):
        for row in m.iter_rows(named=True):
            problem._apply_constraint(
                problem._model.add, "fix", {"var": problem.store.get(row[col]), "value": row["_value"]},
                {k: row[k] for k in keys}, gid, (col,), lambda r: (r["var"] == r["value"],),
                call_id, con_name=f"fix:{label}")
