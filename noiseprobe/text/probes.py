import torch
from typing import Dict, Any
from noiseprobe.noiseprobe.base import BaseProbe

class TextNoiseProbe(BaseProbe):
    name = 'text_noise'
    
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
    
    def __call__(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Add random noise to text embeddings.
        """
        noise = torch.randn_like(X) * self.noise_level
        return X + noise
    
    def get_name(self):
        return self.name

class TextDropoutProbe(BaseProbe):
    name = 'text_dropout'
    
    def __init__(self, dropout_rate: float = 0.1):
        self.dropout_rate = dropout_rate
    
    def __call__(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Randomly zero out some elements in text embeddings.
        """
        mask = torch.rand_like(X) > self.dropout_rate
        return X * mask
    
    def get_name(self):
        return self.name

class TextShuffleProbe(BaseProbe):
    name = 'text_shuffle'
    
    def __init__(self, shuffle_ratio: float = 0.1):
        self.shuffle_ratio = shuffle_ratio
    
    def __call__(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Shuffle a portion of the text embeddings.
        """
        seq_len = X.size(1)
        shuffle_len = int(seq_len * self.shuffle_ratio)
        if shuffle_len > 0:
            start_idx = torch.randint(0, seq_len - shuffle_len, (1,))
            indices = torch.arange(start_idx, start_idx + shuffle_len)
            shuffled_indices = indices[torch.randperm(shuffle_len)]
            X[:, indices] = X[:, shuffled_indices]
        return X 
    
    def get_name(self):
        return self.name