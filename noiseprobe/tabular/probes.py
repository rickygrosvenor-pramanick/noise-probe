import torch
from typing import List
from noiseprobe.noiseprobe.base import BaseProbe


class GaussianNoiseProbeTabular(BaseProbe):
    """
    Adds Gaussian noise to specified columns of a tabular tensor.
    """
    name = 'gaussian_noise_tabular'

    def __call__(
        self,
        X: torch.Tensor,
        columns: List[int],
        std: float = 0.1
    ) -> torch.Tensor:
        X_noisy = X.clone()
        for col in columns:
            noise = torch.randn(X.size(0), device=X.device) * std
            X_noisy[:, col] += noise
        return X_noisy
    
    def get_name(self):
        return 'gaussian_noise_tabular'


class MaskFeaturesProbeTabular(BaseProbe):
    """
    Masks (zeroes out) values in specified columns with a given probability.
    """
    name = 'mask_features_tabular'

    def __call__(
        self,
        X: torch.Tensor,
        columns: List[int],
        mask_prob: float = 0.1
    ) -> torch.Tensor:
        X_masked = X.clone()
        for col in columns:
            mask = torch.rand(X.size(0), device=X.device) < mask_prob
            X_masked[mask, col] = 0.0
        return X_masked
    
    def get_name(self):
        return 'mask_features_tabular'


class FlipCategoriesProbeTabular(BaseProbe):
    """
    Randomly flips categorical values in a specified column.
    """
    name = 'flip_categories_tabular'

    def __call__(
        self,
        X: torch.Tensor,
        column: int,
        num_classes: int,
        flip_prob: float = 0.1
    ) -> torch.Tensor:
        X_flipped = X.clone().long()
        mask = torch.rand(X.size(0), device=X.device) < flip_prob
        new_vals = torch.randint(
            low=0,
            high=num_classes,
            size=(mask.sum().item(),),
            device=X.device
        )
        X_flipped[mask, column] = new_vals
        return X_flipped

    def get_name(self):
        return 'flip_categories_tabular'


class ShuffleColumnProbeTabular(BaseProbe):
    """
    Shuffles the entries of a single column independently of other columns.
    """
    name = 'shuffle_column_tabular'

    def __call__(self, X: torch.Tensor, column: int) -> torch.Tensor:
        X_shuffled = X.clone()
        perm = torch.randperm(X.size(0), device=X.device)
        X_shuffled[:, column] = X_shuffled[perm, column]
        return X_shuffled

    def get_name(self):
        return 'shuffle_column_tabular'
