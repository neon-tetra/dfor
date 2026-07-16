class VarStore:
    def __init__(self):
        self._records = {}        # id(str) -> (var, entity_name, birth_grain)
        self._n = 0

    def next_id(self):
        id_ = f"dforvar_{self._n}"
        self._n += 1
        return id_

    def put(self, id_, var, entity_name, birth_grain):
        self._records[id_] = (var, entity_name, tuple(birth_grain))
        return id_

    def get(self, id_):
        return self._records[id_][0]          # the var object

    def is_id(self, val):
        return isinstance(val, str) and val in self._records

    def birth_grain(self, id_):
        return self._records[id_][2]

    def entity_of(self, id_):
        return self._records[id_][1]


class Entity:
    def __init__(self, name, kind):
        self.name = name          # column name
        self.kind = kind          # "satvar" | "scalar"
        self.join_field = False

    def __repr__(self):
        return f"Entity({self.name!r}, {self.kind!r})"


class Registry:
    def __init__(self):
        self._entities = {}       # column name -> Entity
        self.grains = []          # observed (entity, born, now, folded) sightings

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