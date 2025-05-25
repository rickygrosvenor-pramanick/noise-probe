# Base interface for all robustness testers
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
import torch


# Registry for probe classes - dictionary mapping each probe’s unique name string to its implementing class
typing_probe_registry: Dict[str, Type['BaseProbe']] = {}

# Base class for all robustness testers
class BaseRobustnessTester(ABC):
    @abstractmethod
    def evaluate(self, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, Any]:
        """
        Run the full suite of probes on input data X (and optional labels y).
        Returns a dict mapping each probe name to a (metric_name, metric_value) tuple.
        """
        pass

    @abstractmethod
    def run_probe(self, probe_name: str, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Execute a single probe by name, passing extra arguments.
        """
        pass

# Base class for all probes
class BaseProbe(ABC):
    name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'name') and cls.name:
            typing_probe_registry[cls.name] = cls

    @abstractmethod
    def __call__(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the perturbation to X and return the modified tensor.
        """
        pass




