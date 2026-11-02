"""Public method API; defer imports so shared geometry can import submodules."""
from importlib import import_module

__all__ = ['WorldRoPEDotProductAttention']

############################################################################
# Expose the public method API lazily to avoid cross-package import cycles.
############################################################################
def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(__name__ + ".attention"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
