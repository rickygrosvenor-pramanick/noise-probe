import torch
from typing import Dict, Any
from noiseprobe.noiseprobe.base import BaseProbe

class TimeseriesNoiseProbe(BaseProbe):
    name = 'timeseries_noise'
    
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
    
    def __call__(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Add random noise to timeseries data.
        """
        noise = torch.randn_like(X) * self.noise_level
        return X + noise

class TimeseriesSmoothProbe(BaseProbe):
    name = 'timeseries_smooth'
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
    
    def __call__(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply moving average smoothing to timeseries data.
        """
        # Add padding for the edges
        pad_size = self.window_size // 2
        padded = torch.nn.functional.pad(X, (pad_size, pad_size), mode='replicate')
        
        # Create the smoothing kernel
        kernel = torch.ones(1, 1, self.window_size) / self.window_size
        
        # Apply 1D convolution for smoothing
        smoothed = torch.nn.functional.conv1d(
            padded.unsqueeze(1), 
            kernel.unsqueeze(1),
            padding=0
        ).squeeze(1)
        
        return smoothed

class TimeseriesJitterProbe(BaseProbe):
    name = 'timeseries_jitter'
    
    def __init__(self, jitter_scale: float = 0.1):
        self.jitter_scale = jitter_scale
    
    def __call__(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Add time-domain jitter to timeseries data.
        """
        seq_len = X.size(1)
        jitter = torch.randn_like(X) * self.jitter_scale
        
        # Ensure jitter doesn't cause index out of bounds
        jitter = torch.clamp(jitter, -0.5, 0.5)
        
        # Create time indices
        time_indices = torch.arange(seq_len, device=X.device).float()
        jittered_indices = time_indices + jitter
        
        # Interpolate values at jittered positions
        jittered_X = torch.zeros_like(X)
        for i in range(seq_len):
            jittered_X[:, i] = torch.nn.functional.grid_sample(
                X.unsqueeze(1),
                jittered_indices[:, i].view(-1, 1, 1),
                mode='linear',
                align_corners=True
            ).squeeze(1)
        
        return jittered_X 