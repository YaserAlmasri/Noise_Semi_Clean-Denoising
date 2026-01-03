"""
IDDM Network Architecture (PyTorch Version)

This module implements the three-module architecture for IDDM:
1. Noise Suppressor (NS) - Encoder-Decoder for extracting noise features
2. Signal Preserver (SP) - Siamese network for invariant signal features
3. Fusion Module (FM) - Combines features with time embedding

The complete restorer: R_θ(X_t; X_{t+1}) = FM(NS(X_t), SP(X_t, X_{t+1}), t)

Converted from Keras to PyTorch - same architecture and logic preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# =============================================================================
# Time Embedding
# =============================================================================
def get_time_embedding(t, embedding_dim=128, device=None):
    """
    Sinusoidal time embedding (from Transformer).
    
    Args:
        t: Timestep tensor (batch_size,)
        embedding_dim: Dimension of embedding
        device: Torch device
        
    Returns:
        Time embedding tensor (batch_size, embedding_dim)
    """
    if device is None:
        device = t.device
    
    half_dim = embedding_dim // 2
    
    # Frequencies
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
    )
    
    # Ensure t is float and has proper shape
    t = t.float()
    if len(t.shape) == 0:
        t = t.unsqueeze(0)
    
    angles = t.unsqueeze(-1) * frequencies.unsqueeze(0)
    
    # Concatenate sin and cos
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    return embedding


class TimeEmbeddingLayer(nn.Module):
    """Layer for time embedding with learnable projection."""
    
    def __init__(self, embedding_dim=128, projection_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        self.dense1 = nn.Linear(embedding_dim, embedding_dim)
        self.dense2 = nn.Linear(embedding_dim, projection_dim)
        
    def forward(self, t):
        # Get sinusoidal embedding
        emb = get_time_embedding(t, self.embedding_dim, device=t.device)
        # Project with SiLU activation (swish)
        emb = F.silu(self.dense1(emb))
        emb = self.dense2(emb)
        return emb


# =============================================================================
# Utility Blocks
# =============================================================================
class ConvBlock(nn.Module):
    """Convolution block: Conv -> BN -> LeakyReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
        
        # Projection for channel mismatch
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        shortcut = self.proj(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        
        out = out + shortcut
        out = self.act(out)
        return out


# =============================================================================
# Noise Suppressor (NS) - Encoder-Decoder
# =============================================================================
class NoiseSuppressor(nn.Module):
    """
    Noise Suppressor module - Encoder-Decoder architecture.
    Extracts noise-related features from the input state X_t.
    """
    
    def __init__(self, in_channels=2, base_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1_conv1 = ConvBlock(in_channels, base_filters)
        self.enc1_conv2 = ConvBlock(base_filters, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2_conv1 = ConvBlock(base_filters, base_filters * 2)
        self.enc2_conv2 = ConvBlock(base_filters * 2, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3_conv1 = ConvBlock(base_filters * 2, base_filters * 4)
        self.enc3_conv2 = ConvBlock(base_filters * 4, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck_conv1 = ConvBlock(base_filters * 4, base_filters * 8)
        self.bottleneck_conv2 = ConvBlock(base_filters * 8, base_filters * 8)
        
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3_conv1 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2_conv1 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1_conv1 = ConvBlock(base_filters * 2 + base_filters, base_filters)
        
        # Output
        self.output_conv = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1_conv2(self.enc1_conv1(x))
        p1 = self.pool1(e1)
        
        e2 = self.enc2_conv2(self.enc2_conv1(p1))
        p2 = self.pool2(e2)
        
        e3 = self.enc3_conv2(self.enc3_conv1(p2))
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck_conv2(self.bottleneck_conv1(p3))
        
        # Decoder
        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='nearest')
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_conv1(d3)
        
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='nearest')
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_conv1(d2)
        
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='nearest')
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_conv1(d1)
        
        # Output
        output = self.output_conv(d1)
        return output


# =============================================================================
# Signal Preserver (SP) - Siamese Network
# =============================================================================
class SiameseBranch(nn.Module):
    """Single branch of Siamese network for Signal Preserver."""
    
    def __init__(self, in_channels=2, base_filters=32):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, base_filters)
        self.conv2 = ConvBlock(base_filters, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(base_filters, base_filters * 2)
        self.conv4 = ConvBlock(base_filters * 2, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = ConvBlock(base_filters * 2, base_filters * 4)
        self.conv6 = ConvBlock(base_filters * 4, base_filters * 4)
        
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.pool1(x)
        
        x = self.conv4(self.conv3(x))
        x = self.pool2(x)
        
        x = self.conv6(self.conv5(x))
        return x


class SignalPreserver(nn.Module):
    """
    Signal Preserver module - Siamese Network.
    Extracts invariant signal features from adjacent states (X_t, X_{t+1}).
    """
    
    def __init__(self, in_channels=2, base_filters=32, input_size=78):
        super().__init__()
        
        self.input_size = input_size
        
        # Shared branch (Siamese)
        self.branch = SiameseBranch(in_channels, base_filters)
        
        # Upsampling path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ConvBlock(base_filters * 4, base_filters * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = ConvBlock(base_filters * 2, base_filters)
        
        # Output
        self.output_conv = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        
    def forward(self, x_t, x_t1):
        # Extract features from both inputs using shared weights
        features_t = self.branch(x_t)
        features_t1 = self.branch(x_t1)
        
        # Combine features (average for invariant features)
        combined = (features_t + features_t1) / 2
        
        # Upsample back to original size
        x = self.up1(combined)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = self.up_conv2(x)
        
        # Resize to exact input shape
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode='nearest')
        output = self.output_conv(x)
        
        return output
    
    def get_features(self, x):
        """Get branch features for loss computation."""
        return self.branch(x)


# =============================================================================
# Fusion Module (FM)
# =============================================================================
class FusionModule(nn.Module):
    """
    Fusion Module - Combines NS and SP features with time embedding.
    Uses 5 residual blocks for fusion.
    """
    
    def __init__(self, input_channels=2, ns_filters=32, sp_filters=32, time_dim=64):
        super().__init__()
        
        self.input_channels = input_channels
        filters = 64
        
        # Residual Block 1 - Fuse features
        self.res1 = ResidualBlock(ns_filters + sp_filters, filters)
        
        # Time projection
        self.time_proj = nn.Linear(time_dim, filters)
        
        # Residual Blocks 2-5 with time embedding
        self.res2 = ResidualBlock(filters, filters)
        self.res3 = ResidualBlock(filters, filters)
        self.res4 = ResidualBlock(filters, filters)
        self.res5 = ResidualBlock(filters, filters)
        
        # Final projection to predict noise
        self.final_conv1 = nn.Conv2d(filters, 32, 3, padding=1)
        self.final_act = nn.LeakyReLU(0.1)
        self.noise_output = nn.Conv2d(32, input_channels, 3, padding=1)
        
    def forward(self, ns_features, sp_features, time_emb, original_input):
        # Concatenate NS and SP features
        fused = torch.cat([ns_features, sp_features], dim=1)
        
        # Residual Block 1
        x = self.res1(fused)
        
        # Time embedding to spatial
        time_proj = self.time_proj(time_emb)
        time_spatial = time_proj.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Residual Blocks 2-5 with time embedding
        x = self.res2(x + time_spatial)
        x = self.res3(x + time_spatial)
        x = self.res4(x + time_spatial)
        x = self.res5(x + time_spatial)
        
        # Final projection to predict noise
        x = self.final_act(self.final_conv1(x))
        noise_pred = self.noise_output(x)
        
        # Output: X_{t-1} = X_t - N̂
        output = original_input - noise_pred
        
        return output


# =============================================================================
# Complete IDDM Restorer
# =============================================================================
class IDDMRestorer(nn.Module):
    """
    Complete IDDM Restorer model combining NS, SP, and FM.
    
    R_θ(X_t; X_{t+1}) = FM(NS(X_t), SP(X_t, X_{t+1}), t)
    
    NOTE: Expects input in channels-first format (B, C, H, W)
    """
    
    def __init__(self, input_shape=(78, 78, 2), base_filters=32, time_dim=64):
        super().__init__()
        
        self.input_shape = input_shape
        self.base_filters = base_filters
        self.time_dim = time_dim
        
        # Input channels (last element of shape for channels-last format)
        in_channels = input_shape[-1]
        input_size = input_shape[0]
        
        # Build modules
        self.noise_suppressor = NoiseSuppressor(in_channels, base_filters)
        self.signal_preserver = SignalPreserver(in_channels, base_filters, input_size)
        self.fusion_module = FusionModule(in_channels, base_filters, base_filters, time_dim)
        self.time_embedding = TimeEmbeddingLayer(128, time_dim)
        
    def forward(self, X_t, X_t1, t):
        """
        Forward pass.
        
        Args:
            X_t: Current state (batch, C, H, W) - channels first
            X_t1: Adjacent state (batch, C, H, W)
            t: Timestep (batch,)
            
        Returns:
            X_{t-1}: Denoised state
        """
        # Get features from NS
        ns_features = self.noise_suppressor(X_t)
        
        # Get features from SP
        sp_features = self.signal_preserver(X_t, X_t1)
        
        # Get time embedding
        time_emb = self.time_embedding(t)
        
        # Fuse and predict
        output = self.fusion_module(ns_features, sp_features, time_emb, X_t)
        
        return output
    
    def get_sp_features(self, X):
        """Get Signal Preserver features for loss computation."""
        return self.signal_preserver.get_features(X)


def create_iddm_model(input_shape=(78, 78, 2), base_filters=32, time_dim=64, device=None):
    """
    Create and return IDDM Restorer model.
    
    Args:
        input_shape: Input shape (H, W, C) - channels last format
        base_filters: Base filter count
        time_dim: Time embedding dimension
        device: Torch device
        
    Returns:
        IDDMRestorer model on device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = IDDMRestorer(input_shape, base_filters, time_dim)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"IDDM Model parameters: {total_params:,}")
    
    return model


if __name__ == "__main__":
    print("Testing IDDM Model Architecture (PyTorch)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_iddm_model(input_shape=(78, 78, 2), device=device)
    
    # Test forward pass (channels first for PyTorch)
    batch_size = 4
    X_t = torch.randn(batch_size, 2, 78, 78, device=device)
    X_t1 = torch.randn(batch_size, 2, 78, 78, device=device)
    t = torch.tensor([500, 600, 700, 800], dtype=torch.float32, device=device)
    
    output = model(X_t, X_t1, t)
    
    print(f"Input shape: {X_t.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Model architecture working correctly!")
