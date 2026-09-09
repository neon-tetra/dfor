import polars as pl


class VarStore:
    """Object arena + captured metadata.

    - The var *objects* live in a plain dict keyed by string id (the hot path:
      get()/is_id() are dict ops, called tens of thousands of times).
    - The *metadata* is appended to a list and crystallized to a frame on demand.
    """

    def __init__(self, ids):
        self._ids = ids
        self._objs = {}          # id -> var object   (hot-path arena, dict lookup)
        self._meta = []          # list of dicts      (crystallizes to a frame)
        self._entity_index = {}  # id -> entity, maintained incrementally, O(1) per put
        self._birth_index = {}   # id -> birth_grain_id, same

    def next_id(self):
        return self._ids.next("var")

    def put(self, id_, var, entity_name, birth_grain_id):
        self._objs[id_] = var
        self._meta.append({
            "var_id": id_,
            "entity": entity_name,
            "birth_grain_id": birth_grain_id,
        })
        self._entity_index[id_] = entity_name
        self._birth_index[id_] = birth_grain_id
        return id_

    # ---- hot path: stays dict ops, unchanged semantics ----
    def get(self, id_):
        return self._objs[id_]

    def is_id(self, val):
        return isinstance(val, str) and val in self._objs

    # ---- metadata accessors (used during capture, small/rare) ----
    def entity_of(self, id_):
        return self._entity_index[id_]

    def birth_grain(self, id_):
        return self._birth_index[id_]

    # ---- report-time crystallization ----
    def to_frame(self):
        if not self._meta:
            return pl.DataFrame(
                schema={"var_id": pl.String, "entity": pl.String,
                        "birth_grain_id": pl.String})
        return pl.DataFrame(self._meta)


class Grains:
    """Dict-interned grains. The frozenset of entity names is the natural key;
    membership is the 'have I seen this grain?' test (O(1), legible).

    Capture-time truth lives in the dict; the normalized (grain_id, entity)
    members table is a projection produced at report time.

    Labels are keyed on the same frozenset, so a grain can be named before
    or after it's first seen -- the label attaches whenever the key matches.
    """

    def __init__(self, ids):
        self._ids = ids
        self._by_key = {}    # frozenset(entities) -> grain_id
        self._order = []     # preserves declaration order: [(grain_id, (entities...))]
        self._labels = {}    # frozenset(entities) -> user-facing name

    def id_for(self, entities):
        key = frozenset(entities)
        gid = self._by_key.get(key)
        if gid is None:
            gid = self._ids.next("grain")
            self._by_key[key] = gid
            self._order.append((gid, tuple(entities)))
        return gid

    def label(self, entities, name):
        self._labels[frozenset(entities)] = name

    def entities_of(self, grain_id):
        for gid, ents in self._order:
            if gid == grain_id:
                return ents
        return ()

    # ---- report-time crystallization: long/normalized members table ----
    def to_frame(self):
        rows = [
            {"grain_id": gid, "entity": ent}
            for gid, ents in self._order
            for ent in ents
        ]
        if not rows:
            return pl.DataFrame(schema={"grain_id": pl.String, "entity": pl.String})
        return pl.DataFrame(rows)

    def labels_to_frame(self):
        rows = [
            {"grain_id": gid, "label": self._labels[frozenset(ents)]}
            for gid, ents in self._order
            if frozenset(ents) in self._labels
        ]
        if not rows:
            return pl.DataFrame(schema={"grain_id": pl.String, "label": pl.String})
        return pl.DataFrame(rows)


class Entity:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind          # "satvar" | "scalar"
        self.join_field = False

    def __repr__(self):
        return f"Entity({self.name!r}, {self.kind!r})"


class Registry:
    def __init__(self):
        self._entities = {}       # column name -> Entity
        self.grains = []          # (entity, born, now, folded) sightings

    def add(self, name, kind):
        if name not in self._entities:
            self._entities[name] = Entity(name, kind)
        return self._entities[name]

    def get(self, name):
        return self._entities[name]

    def mark_join_fields(self, names):
        for name in names:
            if name in self._entities:
                self._entities[name].join_field = True

    def record_sighting(self, entity, born, now, folded):
        self.grains.append((entity, born, now, folded))

    def entities_to_frame(self):
        rows = [{"entity": e.name, "kind": e.kind, "join_field": e.join_field}
                for e in self._entities.values()]
        if not rows:
            return pl.DataFrame(
                schema={"entity": pl.String, "kind": pl.String,
                        "join_field": pl.Boolean})
        return pl.DataFrame(rows)


class ConstraintStore:
    def __init__(self, ids):
        self._ids = ids
        self._rows = []
        self._by_litindex = {}
        self._by_name = {}

    def next_name(self):
        return self._ids.next("con")

    def next_call_id(self):
        """One id per *pipe call* (shared by every row it produces), not
        per row -- the signal that tells apart two constraints which share
        a verb name and a grain but come from different `.pipe()` sites."""
        return self._ids.next("call")

    def put(self, name, ctype, grain_id, entities, expr_str, row_keys, call_id,
             con_index, lit_index=None, name_=None, **kwargs):
        rec = {
            "con_id": name, "type": ctype, "grain_id": grain_id,
            "entities": tuple(entities), "expr": expr_str,
            "row": row_keys, "call_id": call_id, "con_index": con_index,
            "lit_index": lit_index, "name": kwargs.get("name", name_),
        }
        self._rows.append(rec)
        self._by_name[name] = rec
        if lit_index is not None:
            self._by_litindex[lit_index] = name
        return name

    def get(self, name):
        return self._by_name[name]

    def name_for_litindex(self, idx):
        return self._by_litindex.get(idx)

    # ---- crystallization: the normalized row-keys long table ----
    def rows_to_frame(self):
        # `value` has to hold whatever a grain column's dtype is -- ints
        # most of the time, but a string-valued grain (a "skill" column,
        # say) is just as legitimate. Building this straight from mixed
        # Python types makes polars infer one dtype from however many rows
        # it samples and then choke the moment a later row disagrees (it
        # picked Int64 from early int-valued keys, then hit a string).
        # Stringifying every value up front sidesteps that -- consumers
        # already have to interpret `value` in light of `key` regardless.
        recs = [
            {"con_id": r["con_id"], "key": k, "value": str(v)}
            for r in self._rows
            for k, v in r["row"].items()
        ]
        if not recs:
            return pl.DataFrame(schema={
                "con_id": pl.String, "key": pl.String, "value": pl.String})
        return pl.DataFrame(recs)

    # ---- report-time crystallization ----
    def to_frame(self):
        if not self._rows:
            return pl.DataFrame(schema={
                "con_id": pl.String, "type": pl.String, "grain_id": pl.String,
                "entities": pl.List(pl.String), "expr": pl.String,
                "call_id": pl.String, "con_index": pl.Int64, "lit_index": pl.Int64,
                "name": pl.String})
        # 'row' (a dict) and 'entities' (a tuple) are nested; keep entities as a
        # list column, and drop the per-row dict from the frame projection
        # (it's kept on the record for point access, but isn't tabular-friendly).
        flat = [{
            "con_id": r["con_id"], "type": r["type"], "grain_id": r["grain_id"],
            "entities": list(r["entities"]), "expr": r["expr"],
            "call_id": r["call_id"], "con_index": r["con_index"],
            "lit_index": r["lit_index"], "name": r["name"],
        } for r in self._rows]
        return pl.DataFrame(flat)
