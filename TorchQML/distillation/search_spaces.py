"""Compatibility aggregator for distillation architecture search spaces."""

from .broad_search import *
from .broad_search import __all__ as _broad_all
from .circuit_ops import *
from .circuit_ops import __all__ as _ops_all
from .flexible_search import *
from .flexible_search import __all__ as _flexible_all
from .repeated_search import *
from .repeated_search import __all__ as _repeated_all


__all__ = [
    *_ops_all,
    *_repeated_all,
    *_flexible_all,
    *_broad_all,
]
