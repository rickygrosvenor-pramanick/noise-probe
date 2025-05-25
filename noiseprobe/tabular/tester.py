from typing import Any, Callable, Dict, Tuple
import torch

import noiseprobe.noiseprobe.registry as registry
from noiseprobe.noiseprobe.base import BaseRobustnessTester
import noiseprobe.tabular.probes as probes


class TabularRobustnessTester(BaseRobustnessTester):
    """
    Implements robustness evaluation for tabular data by applying selected probes
    and computing a user-specified metric on model outputs.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        metric_fn: Callable[..., float],
        metric_name: str,
        X: torch.Tensor,
        y: torch.Tensor = None
    ):
        """
        Args:
            model: Function mapping input X to predictions (Tensor).
            metric_fn: Function(first_preds, perturbed_preds, y?) -> float.
            metric_name: Label for the computed metric.
        """
        self.model = model
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.X = X
        self.y = y

    def evaluate(
        self,
        probe_kwargs: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Tuple[str, float]]:
        """
        Runs specified tabular probes on X, computes baseline vs perturbed predictions,
        and returns a dict mapping each probe name to (metric_name, metric_value).

        Args:
            X: Input tensor of shape (N, D).
            y: Optional labels for supervised metrics.
            probe_kwargs: Mapping from probe_name to its kwargs dict. Only these probes will run.

        Returns:
            Dict mapping probe_name -> (metric_name, metric_value).
        """
        results: Dict[str, Tuple[str, float]] = {}
        baseline_preds = self.model(self.X)
        probe_kwargs = probe_kwargs or {}

        for probe_name, kwargs in probe_kwargs.items():
            try:
                perturbed = self.run_probe(probe_name, self.X, **kwargs)
                perturbed_preds = self.model(perturbed)

                # Compute metric; include labels if provided
                if self.y is not None:
                    val = self.metric_fn(baseline_preds, perturbed_preds, self.y)
                else:
                    val = self.metric_fn(baseline_preds, perturbed_preds)
                results[probe_name] = (self.metric_name, val)
            except Exception:
                results[probe_name] = (self.metric_name, float('nan'))

        return results

    def run_probe(
        self,
        probe_name: str,
        X: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Executes a single named probe on X with given kwargs, returning the perturbed tensor.
        """
        probe_cls = registry.get_probe_cls(probe_name)
        
        probe = probe_cls()
        return probe(X, **kwargs)


if __name__ == '__main__':
    # Example usage with explicit probe arguments
    def dummy_model(x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def l2_metric(a: torch.Tensor, b: torch.Tensor, y=None) -> float:
        return torch.norm(a - b).item()

    X = torch.randn(100, 5)
    tester = TabularRobustnessTester(dummy_model, l2_metric, 'L2Dist', X)
    # Specify which probes to run and their parameters
    probe_configs = {
        'gaussian_noise': {'columns': [0,1], 'std': 0.2},
        'mask_features': {'columns': [2,3], 'mask_prob': 0.3},
        'shuffle_column': {'column': 4},
        'flip_categories': {'column': 4, 'num_classes': 10, 'flip_prob': 0.2}
    }
    results = tester.evaluate(probe_kwargs=probe_configs)
    print(results)
