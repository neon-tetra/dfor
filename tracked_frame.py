import polars as pl

_TRACKED_METHODS = (
    "pipe",
    "select",
    "filter",
    "group_by",
    "sort",
    "join",
    "with_columns",
    "explode",
    "implode",
    "rename",
    "pivot",
)


class TrackedFrame:
    def __init__(self, frame: pl.DataFrame):
        self._frame = frame

    def __getattr__(self, name):
        if name not in _TRACKED_METHODS:
            raise AttributeError(name)

        def _tracked(*args, **kwargs):
            # log stuff here
            self._frame = getattr(self._frame, name)(*args, **kwargs)
            return self

        return _tracked