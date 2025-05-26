import torch
from typing import Dict, Any, List
from noiseprobe.noiseprobe.base import BaseRobustnessTester
from noiseprobe.noiseprobe.registry import get_probe_cls, list_probes
# Import probes to ensure they get registered
import noiseprobe.timeseries.probes as probes

class TimeseriesRobustnessTester(BaseRobustnessTester):
    def __init__(self, probes: List[str] = None):
        """
        Initialize the timeseries robustness tester with specified probes.
        If no probes are specified, all available timeseries probes will be used.
        """
        if probes is None:
            self.probes = [p for p in list_probes() if p.startswith('timeseries_')]
        else:
            self.probes = probes

    def evaluate(self, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, Any]:
        """
        Run all timeseries probes on the input data and return metrics.
        """
        results = {}
        for probe_name in self.probes:
            probe = get_probe_cls(probe_name)()
            perturbed_X = probe(X)
            
            # Calculate basic metrics
            diff = torch.abs(X - perturbed_X)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            
            # Calculate timeseries-specific metrics
            freq_diff = torch.fft.fft(perturbed_X) - torch.fft.fft(X)
            freq_magnitude = torch.abs(freq_diff).mean().item()
            
            results[probe_name] = {
                'mean_difference': mean_diff,
                'max_difference': max_diff,
                'frequency_difference': freq_magnitude
            }
            
            if y is not None:
                # Add task-specific metrics here if needed
                pass
                
        return results

    def run_probe(self, probe_name: str, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run a single timeseries probe on the input data.
        """
        probe = get_probe_cls(probe_name)(**kwargs)
        return probe(X)

if __name__ == '__main__':
    # Example usage with timeseries data
    # Create a batch of random timeseries data (batch_size, sequence_length, features)
    batch_size = 8
    seq_length = 100
    n_features = 3
    X = torch.randn(batch_size, seq_length, n_features)
    
    # Initialize the tester
    tester = TimeseriesRobustnessTester()
    
    # Run evaluation with all available timeseries probes
    results = tester.evaluate(X)
    
    # Print results
    print("Timeseries Probe Evaluation Results:")
    for probe_name, metrics in results.items():
        print(f"\n{probe_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Example of running a single probe with custom parameters
    print("\nRunning single probe with custom parameters:")
    custom_probe = tester.run_probe('timeseries_smooth', X, window_size=5)
    diff = torch.abs(X - custom_probe)
    print(f"Custom smooth probe - Mean difference: {diff.mean().item():.4f}")
    
    # Example of frequency analysis
    print("\nFrequency analysis of jittered data:")
    jittered = tester.run_probe('timeseries_jitter', X, jitter_scale=0.1)
    freq_diff = torch.fft.fft(jittered) - torch.fft.fft(X)
    freq_magnitude = torch.abs(freq_diff).mean().item()
    print(f"Frequency magnitude difference: {freq_magnitude:.4f}") 