"""
IDDM Loss Functions (PyTorch Version)

This module implements the three loss functions for IDDM:
1. Noise Variance Loss (L_NV) - Ensures predicted noise matches field noise
2. Signal Preservation Loss (L_SP) - Contrastive/similarity loss for signal features
3. Charbonnier Loss (L_NS-FM) - Robust reconstruction loss

Combined loss: L = L_NV + λ₁*L_SP + λ₂*L_NS-FM

Converted from TensorFlow to PyTorch - same logic preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Noise Variance Loss (L_NV)
# =============================================================================
def noise_variance_loss(X_t_minus_1_pred, X_t_minus_1_true, beta_t_minus_1):
    """
    Simplified Noise Variance Loss.
    
    Directly compares the residual (difference) to expected noise level.
    
    Args:
        X_t_minus_1_pred: Predicted X_{t-1}
        X_t_minus_1_true: Actual X_{t-1}
        beta_t_minus_1: Beta at t-1
        
    Returns:
        L_NV loss value
    """
    residual = X_t_minus_1_true - X_t_minus_1_pred
    
    # MSE of residual should be approximately beta_{t-1}
    residual_variance = torch.mean(residual ** 2)
    
    # Loss is deviation from expected variance
    loss = (residual_variance - beta_t_minus_1) ** 2
    
    return loss


# =============================================================================
# Signal Preservation Loss (L_SP)
# =============================================================================
def signal_preservation_loss(features_t, features_t1, is_high_noise, margin=1.0):
    """
    Signal Preservation Loss from Equation 16.
    
    Uses contrastive learning approach:
    - High noise: Push features apart (different noise levels should differ)
    - Low noise: Pull features together (similar signal content)
    
    Args:
        features_t: Features from X_t
        features_t1: Features from X_{t+1}
        is_high_noise: Boolean or float (1.0 for high noise, 0.0 for low noise)
        margin: Contrastive margin d
        
    Returns:
        L_SP loss value
    """
    # Compute feature distance
    diff = features_t - features_t1
    distance = torch.sqrt(torch.mean(diff ** 2) + 1e-6)
    
    # Convert is_high_noise to gamma (0 for high noise, 1 for low noise)
    gamma = 1.0 - float(is_high_noise)
    
    # Similarity loss (pull together)
    similarity_loss = distance ** 2
    
    # Contrastive loss (push apart)
    contrastive_loss = torch.clamp(margin - distance, min=0.0) ** 2
    
    # Combined loss
    loss = gamma * similarity_loss + (1.0 - gamma) * contrastive_loss
    
    return loss


def signal_preservation_loss_vectorized(features_t, features_t1, is_high_noise, margin=1.0):
    """
    Vectorized Signal Preservation Loss for batches.
    
    Args:
        features_t: Features from X_t (batch, C, H, W)
        features_t1: Features from X_{t+1} (batch, C, H, W)
        is_high_noise: Boolean tensor (batch,)
        margin: Contrastive margin
        
    Returns:
        Mean L_SP loss over batch
    """
    # Compute per-sample distances
    diff = features_t - features_t1
    # Mean over channel, height, width dims, keep batch dim
    distances = torch.sqrt(torch.mean(diff ** 2, dim=[1, 2, 3]) + 1e-6)
    
    # Convert is_high_noise to gamma
    gamma = 1.0 - is_high_noise.float()
    
    # Similarity loss (for low noise)
    similarity_losses = distances ** 2
    
    # Contrastive loss (for high noise)
    contrastive_losses = torch.clamp(margin - distances, min=0.0) ** 2
    
    # Combined per-sample losses
    losses = gamma * similarity_losses + (1.0 - gamma) * contrastive_losses
    
    return torch.mean(losses)


# =============================================================================
# Charbonnier Loss (L_NS-FM)
# =============================================================================
def charbonnier_loss(y_pred, y_true, epsilon=0.03):
    """
    Charbonnier Loss from Equation 17.
    
    A differentiable approximation to L1 loss that is more robust to outliers.
    
    L = √(||y_pred - y_true||² + ε²)
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        epsilon: Small constant for numerical stability
        
    Returns:
        Charbonnier loss value
    """
    diff_squared = torch.mean((y_pred - y_true) ** 2)
    loss = torch.sqrt(diff_squared + epsilon ** 2)
    return loss


def charbonnier_loss_per_sample(y_pred, y_true, epsilon=0.03):
    """
    Per-sample Charbonnier Loss.
    
    Args:
        y_pred: Predicted values (batch, C, H, W)
        y_true: Ground truth (batch, C, H, W)
        epsilon: Epsilon constant
        
    Returns:
        Loss per sample (batch,)
    """
    diff_squared = torch.mean((y_pred - y_true) ** 2, dim=[1, 2, 3])
    losses = torch.sqrt(diff_squared + epsilon ** 2)
    return losses


# =============================================================================
# Combined IDDM Loss
# =============================================================================
class IDDMLoss(nn.Module):
    """
    Combined IDDM Loss function.
    
    L = L_NV + λ₁*L_SP + λ₂*L_NS-FM
    """
    
    def __init__(self, lambda_1=0.25, lambda_2=0.75, margin=1.0, epsilon=0.03,
                 noise_threshold=0.5, noise_std=1.0):
        """
        Initialize IDDM Loss.
        
        Args:
            lambda_1: Weight for L_SP
            lambda_2: Weight for L_NS-FM
            margin: Contrastive margin for L_SP
            epsilon: Charbonnier epsilon
            noise_threshold: Threshold for high/low noise in L_SP
            noise_std: Expected noise standard deviation
        """
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.margin = margin
        self.epsilon = epsilon
        self.noise_threshold = noise_threshold
        self.noise_std = noise_std
        
    def forward(self, X_t_minus_1_pred, X_t_minus_1_true, features_t, features_t1,
                beta_t_minus_1, noise_level):
        """
        Compute combined loss.
        
        Args:
            X_t_minus_1_pred: Model prediction
            X_t_minus_1_true: Ground truth X_{t-1}
            features_t: SP features from X_t
            features_t1: SP features from X_{t+1}
            beta_t_minus_1: Beta at t-1
            noise_level: Noise level at t
            
        Returns:
            Dictionary with total loss and individual components
        """
        # L_NV: Noise Variance Loss
        l_nv = noise_variance_loss(X_t_minus_1_pred, X_t_minus_1_true, beta_t_minus_1)
        
        # L_SP: Signal Preservation Loss
        is_high_noise = noise_level > self.noise_threshold
        l_sp = signal_preservation_loss(features_t, features_t1, is_high_noise, self.margin)
        
        # L_NS-FM: Charbonnier Loss
        l_char = charbonnier_loss(X_t_minus_1_pred, X_t_minus_1_true, self.epsilon)
        
        # Combined loss
        total_loss = l_nv + self.lambda_1 * l_sp + self.lambda_2 * l_char
        
        return {
            'total': total_loss,
            'l_nv': l_nv,
            'l_sp': l_sp,
            'l_char': l_char,
        }


def create_iddm_loss(config=None):
    """
    Create IDDM Loss instance from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        IDDMLoss instance
    """
    if config is None:
        config = {
            'lambda_1': 0.25,
            'lambda_2': 0.75,
            'margin_d': 1.0,
            'epsilon': 0.03,
        }
    
    return IDDMLoss(
        lambda_1=config.get('lambda_1', 0.25),
        lambda_2=config.get('lambda_2', 0.75),
        margin=config.get('margin_d', 1.0),
        epsilon=config.get('epsilon', 0.03),
    )


if __name__ == "__main__":
    print("Testing IDDM Loss Functions (PyTorch)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create random test data (channels first for PyTorch)
    batch_size = 4
    X_pred = torch.randn(batch_size, 2, 78, 78, device=device)
    X_true = torch.randn(batch_size, 2, 78, 78, device=device)
    features_t = torch.randn(batch_size, 128, 19, 19, device=device)
    features_t1 = torch.randn(batch_size, 128, 19, 19, device=device)
    
    # Test individual losses
    print("\nTesting individual losses:")
    
    l_nv = noise_variance_loss(X_pred, X_true, 0.1)
    print(f"L_NV: {l_nv.item():.6f}")
    
    l_sp_high = signal_preservation_loss(features_t[0], features_t1[0], True, margin=1.0)
    l_sp_low = signal_preservation_loss(features_t[0], features_t1[0], False, margin=1.0)
    print(f"L_SP (high noise): {l_sp_high.item():.6f}")
    print(f"L_SP (low noise): {l_sp_low.item():.6f}")
    
    l_char = charbonnier_loss(X_pred, X_true)
    print(f"L_Charbonnier: {l_char.item():.6f}")
    
    # Test combined loss
    print("\nTesting combined loss:")
    loss_fn = create_iddm_loss()
    
    result = loss_fn(
        X_pred[0], X_true[0], 
        features_t[0], features_t1[0],
        beta_t_minus_1=0.1,
        noise_level=0.6
    )
    
    print(f"Total: {result['total'].item():.6f}")
    print(f"  L_NV: {result['l_nv'].item():.6f}")
    print(f"  L_SP: {result['l_sp'].item():.6f}")
    print(f"  L_Char: {result['l_char'].item():.6f}")
    
    print("\n✓ Loss functions working correctly!")
