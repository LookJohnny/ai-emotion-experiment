from .probe import ProbeSpec
from .pack import Pack, CONCEPTS
from .policy import TwoTier, percentile


def Engine(*a, **kw):                      # lazy: importing torch/transformers only when an engine is actually built
    from .engine import Engine as _E
    return _E(*a, **kw)
