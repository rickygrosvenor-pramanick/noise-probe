from typing import Any, Callable, Dict, Tuple
import torch

import noiseprobe.noiseprobe.registry as registry
from noiseprobe.noiseprobe.base import BaseRobustnessTester
# Do not remove this import, it populates the registry dictionary
import noiseprobe.image.probes as probes

class ImageRobustnessTester(BaseRobustnessTester):
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor],
                 metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
                 metric_name: str,
                 X: torch.Tensor,
                 y: torch.Tensor = None):
        
        self.model = model
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.X = X
        self.y = y

    def evaluate(self, probe_kwargs: Dict[str, Dict[str, Any]] = None) -> Dict[str, Tuple[str, float]]:
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
    
    def run_probe(self, probe_name: str, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        probe_cls = registry.get_probe_cls(probe_name)
        probe = probe_cls()
        return probe(X, **kwargs)

if __name__ == '__main__':
    import torchvision.models as models
    import torchvision.transforms.functional as TF
    import cv2

    # 1) Define a dummy model (ResNet18)
    cnn = models.resnet18(pretrained=False)
    cnn.eval()
    model = lambda x: cnn(x)

    # 2) Define a simple L2 metric on logits
    def l2_metric(base_preds: torch.Tensor, perturbed_preds: torch.Tensor) -> float:
        return (base_preds - perturbed_preds).pow(2).sum(dim=1).mean().item()

    # 3) Create a batch of random images (B,3,224,224)
    X = torch.rand(8, 3, 224, 224)

    # 4) Instantiate the tester
    tester = ImageRobustnessTester(
        model=model,
        metric_fn=l2_metric,
        metric_name='mean_l2',
        X=X
    )

    # 5) Specify probes and their parameters
    probe_kwargs = {
        'gaussian_noise_image':         {'std': 0.2},
        'salt_and_pepper_noise_image':  {'prob': 0.1},
        'rotate_image':                 {'angle': 10},
        'jpeg_compression_image':       {'quality': 30},
    }

    # 6) Run evaluation
    results = tester.evaluate(probe_kwargs)

    # 7) Print results
    print("Probe Evaluation Results:")
    for probe, (metric, val) in results.items():
        print(f"{probe:25s} -> {metric}: {val:.4f}")
    