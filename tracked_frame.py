import polars as pl

_TRACKED_METHODS = (
    "pipe", "select", "filter", "group_by", "agg", "sort",
    "join", "with_columns", "explode", "implode", "rename", "pivot",
)


class TrackedFrame:
    def __init__(self, frame: pl.DataFrame, problem):
        self._frame = frame
        self._problem = problem

    def __getattr__(self, name):
        if name not in _TRACKED_METHODS:
            raise AttributeError(name)

        def _tracked(*args, **kwargs):
            result = getattr(self._frame, name)(*args, **kwargs)
            if isinstance(result, TrackedFrame):
                result = result._frame           # unwrap nesting
            self._frame = result
            if isinstance(self._frame, pl.DataFrame):
                self._problem.observe(self._frame)
            return self
        return _tracked