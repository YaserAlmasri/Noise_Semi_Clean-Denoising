"""
U-Net Model for Signal Denoising (PyTorch Version)

This module provides:
- U-Net architecture with skip connections for denoising STFT coefficients
- Custom loss function (MSE + L1 regularization)
- Training utilities

Converted from Keras to PyTorch - same architecture and logic preserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


# =============================================================================
# Device Configuration
# =============================================================================
def get_device():
    """Detect and return available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GPU found: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('No GPU found, using CPU')
    return device


# =============================================================================
# Custom Loss Function
# =============================================================================
def custom_loss(y_pred, y_true, mse_weight=0.9, l1_weight=0.1):
    """
    Custom loss combining MSE reconstruction loss with L1 regularization.
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        mse_weight: Weight for MSE loss term
        l1_weight: Weight for L1 regularization term
        
    Returns:
        Combined loss value
    """
    mse_loss = F.mse_loss(y_pred, y_true)
    l1_norm = torch.mean(torch.abs(y_pred))
    total_loss = mse_weight * mse_loss + l1_weight * l1_norm
    return total_loss


# =============================================================================
# U-Net Architecture (Paper Version)
# =============================================================================
class UNet(nn.Module):
    """
    U-Net model for STFT coefficient denoising - PAPER VERSION.
    
    Architecture from paper:
    - Encoder: 4 levels (64→128→256 filters) + Bottleneck (512)
    - 1 Conv2D + LeakyReLU per encoder level
    - Decoder: 4 levels with Conv2DTranspose + Concat + Conv2D + Dropout
    - Output: Conv2DTranspose with linear activation
    
    Input shape: (batch, 2, 78, 78) - STFT with real/imag channels
    Output shape: (batch, 2, 78, 78) - Denoised STFT
    """
    
    def __init__(self, in_channels=2, out_channels=2, dropout_rate=0.3):
        super(UNet, self).__init__()
        
        # Encoder Block 1: 64 filters
        self.enc1_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Encoder Block 2: 128 filters
        self.enc2_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Encoder Block 3: 256 filters
        self.enc3_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Encoder Block 4: 256 filters (additional level for 78x78 input)
        self.enc4_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck: 512 filters
        self.bottleneck_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Decoder Block 1: 256 filters
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 256 + 256 = 512
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Decoder Block 2: 256 filters
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 256 + 256 = 512
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Decoder Block 3: 128 filters
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 128 + 128 = 256
        
        # Decoder Block 4: 64 filters
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 64 + 64 = 128
        
        # Output layer: Conv2DTranspose with linear activation
        self.output_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=3, padding=1)
        
        # Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        # Input: (batch, 2, H, W) - channels first for PyTorch
        
        # Pad to make divisible by 16 (for 4 pooling layers)
        # Original size is 78x78, pad to 80x80
        x = F.pad(x, (1, 1, 1, 1))  # pad: (left, right, top, bottom)
        
        # Encoder Block 1: 80x80 -> 40x40
        enc1 = self.leaky_relu(self.enc1_conv(x))
        pool1 = self.pool1(enc1)
        
        # Encoder Block 2: 40x40 -> 20x20
        enc2 = self.leaky_relu(self.enc2_conv(pool1))
        pool2 = self.pool2(enc2)
        
        # Encoder Block 3: 20x20 -> 10x10
        enc3 = self.leaky_relu(self.enc3_conv(pool2))
        pool3 = self.pool3(enc3)
        
        # Encoder Block 4: 10x10 -> 5x5
        enc4 = self.leaky_relu(self.enc4_conv(pool3))
        pool4 = self.pool4(enc4)
        
        # Bottleneck: 5x5
        bottleneck = self.leaky_relu(self.bottleneck_conv(pool4))
        
        # Decoder Block 1: 5x5 -> 10x10
        up1 = self.leaky_relu(self.up1(bottleneck))
        up1 = torch.cat([up1, enc4], dim=1)
        dec1 = self.leaky_relu(self.dec1_conv(up1))
        dec1 = self.dropout1(dec1)
        
        # Decoder Block 2: 10x10 -> 20x20
        up2 = self.leaky_relu(self.up2(dec1))
        up2 = torch.cat([up2, enc3], dim=1)
        dec2 = self.leaky_relu(self.dec2_conv(up2))
        dec2 = self.dropout2(dec2)
        
        # Decoder Block 3: 20x20 -> 40x40
        up3 = self.leaky_relu(self.up3(dec2))
        up3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.leaky_relu(self.dec3_conv(up3))
        
        # Decoder Block 4: 40x40 -> 80x80
        up4 = self.leaky_relu(self.up4(dec3))
        up4 = torch.cat([up4, enc1], dim=1)
        dec4 = self.leaky_relu(self.dec4_conv(up4))
        
        # Output (linear activation for regression)
        output = self.output_conv(dec4)
        
        # Crop back to original size (80 -> 78)
        output = output[:, :, 1:-1, 1:-1]
        
        return output


# =============================================================================
# U-Net Architecture (Double Conv2D Version - Alternative)
# =============================================================================
class UNetDoubleConv2D(nn.Module):
    """
    Alternative U-Net model with 2x Conv2D per level (smaller, 3 levels, 16→32→64→128 filters).
    Use this as an alternative to the paper version.
    """
    
    def __init__(self, in_channels=2, out_channels=2):
        super(UNetDoubleConv2D, self).__init__()
        
        # Encoder Block 1
        self.enc1_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.enc1_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Encoder Block 2
        self.enc2_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Encoder Block 3
        self.enc3_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Decoder Block 1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Decoder Block 2
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Decoder Block 3
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec3_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # Output layer
        self.output_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        
        enc1 = self.leaky_relu(self.enc1_conv1(x))
        enc1 = self.leaky_relu(self.enc1_conv2(enc1))
        pool1 = self.pool1(enc1)
        
        enc2 = self.leaky_relu(self.enc2_conv1(pool1))
        enc2 = self.leaky_relu(self.enc2_conv2(enc2))
        pool2 = self.pool2(enc2)
        
        enc3 = self.leaky_relu(self.enc3_conv1(pool2))
        enc3 = self.leaky_relu(self.enc3_conv2(enc3))
        pool3 = self.pool3(enc3)
        
        bottleneck = self.leaky_relu(self.bottleneck_conv1(pool3))
        bottleneck = self.leaky_relu(self.bottleneck_conv2(bottleneck))
        
        up1 = self.leaky_relu(self.up1(bottleneck))
        up1 = torch.cat([up1, enc3], dim=1)
        dec1 = self.leaky_relu(self.dec1_conv1(up1))
        dec1 = self.leaky_relu(self.dec1_conv2(dec1))
        
        up2 = self.leaky_relu(self.up2(dec1))
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.leaky_relu(self.dec2_conv1(up2))
        dec2 = self.leaky_relu(self.dec2_conv2(dec2))
        
        up3 = self.leaky_relu(self.up3(dec2))
        up3 = torch.cat([up3, enc1], dim=1)
        dec3 = self.leaky_relu(self.dec3_conv1(up3))
        dec3 = self.leaky_relu(self.dec3_conv2(dec3))
        
        output = self.output_conv(dec3)
        output = output[:, :, 1:-1, 1:-1]
        
        return output


# =============================================================================
# Convenience Functions
# =============================================================================
def build_unet(input_shape=(78, 78, 2), use_original=False):
    """
    Build U-Net model for STFT coefficient denoising.
    
    Args:
        input_shape: Shape of input tensor (height, width, channels)
        use_original: If True, use smaller original model; else use paper version
                     
    Returns:
        UNet model instance
    """
    in_channels = input_shape[2] if len(input_shape) == 3 else input_shape[0]
    
    if use_original:
        model = UNetDoubleConv2D(in_channels=in_channels, out_channels=in_channels)
        print("Using DOUBLE CONV2D U-Net (3 levels, 2x Conv per level)")
    else:
        model = UNet(in_channels=in_channels, out_channels=in_channels)
        print("Using PAPER U-Net (4 levels, 64→128→256→512)")
    
    return model


def create_and_compile_model(input_shape=(78, 78, 2), learning_rate=0.001, device=None):
    """
    Create U-Net model and optimizer.
    
    Args:
        input_shape: Shape of input tensor
        learning_rate: Initial learning rate for Adam optimizer
        device: Torch device (GPU/CPU)
        
    Returns:
        Tuple of (model, optimizer, device)
    """
    if device is None:
        device = get_device()
    
    model = build_unet(input_shape)
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, optimizer, device


def predict_and_reconstruct(model, X_test, device, batch_size=64):
    """
    Predict denoised STFT and reconstruct complex coefficients.
    
    Args:
        model: Trained PyTorch model
        X_test: Test input (noisy STFT coefficients with stacked real/imag)
                Shape: (N, H, W, 2) - channels last format from numpy
        device: Torch device
        batch_size: Batch size for inference
        
    Returns:
        Complex STFT coefficients
    """
    model.eval()
    
    # Convert to channels-first for PyTorch: (N, H, W, 2) -> (N, 2, H, W)
    X_test_torch = np.transpose(X_test, (0, 3, 1, 2))
    
    decoded_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_test_torch), batch_size):
            batch = torch.tensor(X_test_torch[i:i+batch_size], dtype=torch.float32).to(device)
            output = model(batch)
            decoded_list.append(output.cpu().numpy())
    
    decoded_data = np.concatenate(decoded_list, axis=0)
    
    # Convert back to channels-last: (N, 2, H, W) -> (N, H, W, 2)
    decoded_data = np.transpose(decoded_data, (0, 2, 3, 1))
    
    # Extract real and imaginary parts
    decoded_real = decoded_data[:, :, :, 0]
    decoded_imag = decoded_data[:, :, :, 1]
    
    # Combine into complex array
    decoded_complex = decoded_real + 1j * decoded_imag
    
    print(f"Reconstructed complex STFT shape: {decoded_complex.shape}")
    return decoded_complex


# =============================================================================
# Training Utilities
# =============================================================================
def train_model(model, optimizer, X_train, y_train, X_val, y_val, device,
                epochs=30, batch_size=128, patience=10,
                checkpoint_path='checkpoints/best_model.pth'):
    """
    Train the U-Net model with early stopping and checkpoint saving.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        X_train: Training input (N, H, W, 2) - channels last
        y_train: Training target (N, H, W, 2)
        X_val: Validation input
        y_val: Validation target
        device: Torch device
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        patience: Early stopping patience
        checkpoint_path: Path to save best model checkpoint
        
    Returns:
        Training history
    """
    import os
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Convert to channels-first: (N, H, W, 2) -> (N, 2, H, W)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    y_train = np.transpose(y_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))
    y_val = np.transpose(y_val, (0, 3, 1, 2))
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    n_train = len(X_train)
    n_batches = n_train // batch_size
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Shuffle training data
        indices = np.random.permutation(n_train)
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            idx = indices[start:end]
            
            X_batch = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_train[idx], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = custom_loss(output, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
                y_batch = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(device)
                
                output = model(X_batch)
                loss = custom_loss(output, y_batch)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step()
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            marker = " *"
        else:
            patience_counter += 1
            marker = ""
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}{marker}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"\nBest model saved to: {checkpoint_path}")
    
    return history
