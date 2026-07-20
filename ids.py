from itertools import count


class Ids:
    """One independent monotonic counter per namespace.

    ids.next("var") -> 'var_0', 'var_1', ...
    ids.next("con") -> 'con_0', ...
    Strings are intentional: the prefix tells you the *kind* at a glance,
    and (per design) string ids are load-bearing for satvar detection.
    """

    def __init__(self):
        self._counters = {}

    def next(self, namespace):
        c = self._counters.setdefault(namespace, count())
        return f"{namespace}_{next(c)}"