from typing import Dict, Type, List
from .base import BaseProbe

# Global registry mapping each probe’s name to its implementing class
probe_registry: Dict[str, Type[BaseProbe]] = {}

def register_probe(cls: Type[BaseProbe]) -> Type[BaseProbe]:
    """
    Class decorator to register a new BaseProbe subclass.

    Usage:
        @register_probe
        class MyProbe(BaseProbe):
            name = 'my_probe'
            ...
    """
    name = getattr(cls, 'name', None)
    if not name:
        raise ValueError(f"Cannot register probe without a non-empty 'name': {cls}")
    probe_registry[name] = cls
    return cls

def get_probe_cls(name: str) -> Type[BaseProbe]:
    """
    Retrieve the probe class by its registered name.

    Raises KeyError if not found.
    """
    try:
        return probe_registry[name]
    except KeyError as e:
        raise KeyError(f"Probe '{name}' not found. Available: {list(probe_registry.keys())}") from e

def list_probes() -> List[str]:
    """
    Return a list of all registered probe names.
    """
    return list(probe_registry.keys())
