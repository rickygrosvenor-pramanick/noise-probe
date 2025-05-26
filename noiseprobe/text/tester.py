import torch
from typing import Dict, Any, List
from noiseprobe.noiseprobe.base import BaseRobustnessTester
from noiseprobe.noiseprobe.registry import get_probe_cls, list_probes
# Import probes to ensure they get registered
import noiseprobe.text.probes as probes

class TextRobustnessTester(BaseRobustnessTester):
    def __init__(self, probes: List[str] = None):
        """
        Initialize the text robustness tester with specified probes.
        If no probes are specified, all available text probes will be used.
        """
        if probes is None:
            self.probes = [p for p in list_probes() if p.startswith('text_')]
        else:
            self.probes = probes

    def evaluate(self, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, Any]:
        """
        Run all text probes on the input data and return metrics.
        """
        results = {}
        for probe_name in self.probes:
            probe = get_probe_cls(probe_name)()
            perturbed_X = probe(X)
            
            # Calculate basic metrics
            diff = torch.abs(X - perturbed_X)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            
            results[probe_name] = {
                'mean_difference': mean_diff,
                'max_difference': max_diff
            }
            
            if y is not None:
                # Add task-specific metrics here if needed
                pass
                
        return results

    def run_probe(self, probe_name: str, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run a single text probe on the input data.
        """
        probe = get_probe_cls(probe_name)(**kwargs)
        return probe(X)

if __name__ == '__main__':
    # Example usage with text data
    # Create a batch of random text embeddings (batch_size, sequence_length, embedding_dim)
    batch_size = 8
    seq_length = 32
    embedding_dim = 768  # Typical BERT embedding dimension
    X = torch.randn(batch_size, seq_length, embedding_dim)
    
    # Initialize the tester
    tester = TextRobustnessTester()
    
    # Run evaluation with all available text probes
    results = tester.evaluate(X)
    
    # Print results
    print("Text Probe Evaluation Results:")
    for probe_name, metrics in results.items():
        print(f"\n{probe_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Example of running a single probe with custom parameters
    print("\nRunning single probe with custom parameters:")
    custom_probe = tester.run_probe('text_noise', X, noise_level=0.2)
    diff = torch.abs(X - custom_probe)
    print(f"Custom noise probe - Mean difference: {diff.mean().item():.4f}") 