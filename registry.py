from typing import NamedTuple


class VarStore:
    def __init__(self):
        self._by_id = []          # id -> var object (index is the id)
        self._id_by_var = {}      # var object -> id

    def add(self, var):
        id_ = len(self._by_id)
        self._by_id.append(var)
        self._id_by_var[var] = id_
        return id_

    def get(self, id_):
        return self._by_id[id_]

    def id_of(self, var):
        return self._id_by_var[var]


class Entity:
    def __init__(self, name, kind):
        self.name = name          # column name
        self.kind = kind          # "satvar" | "scalar"
        self.join_field = False


class Registry:
    def __init__(self):
        self._entities = {}       # column name -> Entity

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