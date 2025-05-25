import torch

def add_gaussian_noise(X, columns, std=0.1):
    """
    Add Gaussian noise to selected columns of X.
    
    Args:
        X: Tensor of shape (N, D)
        columns: List of integer column indices
        std: Standard deviation of the noise
    
    Returns:
        X_noisy: Tensor of shape (N, D)
    """
    X_noisy = X.clone()
    for col in columns:
        # Noise vector of size Nx1
        noise = torch.randn(X.size(0)) * std
        X_noisy[:, col] += noise
    return X_noisy


def mask_features(X, columns, mask_prob=0.1):
    """
    Randomly mask (zero out) selected features with probability `mask_prob`.
    
    Args:
        X: Tensor of shape (N, D)
        columns: List of integer column indices
        mask_prob: Probability of masking a given feature value
    
    Returns:
        X_masked: Tensor of shape (N, D)
    """
    X_masked = X.clone()
    for col in columns:
        mask = torch.rand(X.size(0)) < mask_prob
        X_masked[mask, col] = 0.0  # Or `float('nan')` if you handle it downstream
    return X_masked


def flip_categories(X, column, num_classes, flip_prob=0.1):
    """
    Randomly flip categorical values in a given column.
    
    Args:
        X: LongTensor of shape (N, D)
        column: Integer index of the column
        num_classes: Total number of possible classes
        flip_prob: Probability to flip each sample
    
    Returns:
        X_flipped: LongTensor of shape (N, D)
    """
    X_flipped = X.clone()
    mask = torch.rand(X.size(0)) < flip_prob
    flipped_values = torch.randint(0, num_classes, size=(mask.sum(),))
    X_flipped[mask, column] = flipped_values
    return X_flipped


def shuffle_column(X, column):
    """
    Shuffle a column independently of others.
    
    Args:
        X: Tensor of shape (N, D)
        column: Integer index
    
    Returns:
        X_shuffled: Tensor of shape (N, D)
    """
    X_shuffled = X.clone()
    permuted = X_shuffled[:, column][torch.randperm(X.size(0))]
    X_shuffled[:, column] = permuted
    return X_shuffled

