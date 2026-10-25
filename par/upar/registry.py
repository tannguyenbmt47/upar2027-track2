"""Name -> class registry so components are selected from the command line."""


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, name):
        def deco(cls):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' registered twice")
            self._items[name] = cls
            return cls
        return deco

    def build(self, name, *args, **kwargs):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}', available: {sorted(self._items)}")
        return self._items[name](*args, **kwargs)

    def names(self):
        return sorted(self._items)


def parse_kv(items):
    """['gamma=2', 'weighting=deepmar'] -> {'gamma': 2.0, 'weighting': 'deepmar'} (numbers become float/int)."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        out[k] = v
    return out
