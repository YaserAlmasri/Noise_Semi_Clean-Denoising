"""
IDDM Benchmark Training Script (PyTorch Version)

This script implements the IDDM (Iterative Diffusion-based Denoising Model) 
architecture for benchmarking purposes.

NOTE: The original IDDM paper describes a self-supervised two-stage training 
approach (Learning Stage + Restoring Stage) where the model learns to reverse 
diffusion using only noisy data. In our experiments, this pure self-supervised 
approach led to training instability and degraded results (e.g., -27 dB SNR 
instead of positive improvement).

As a practical modification, we train the IDDM architecture using SVD-generated 
semi-clean signals as soft targets. This maintains the benchmark's independence 
from ground truth clean signals (since SVD generates targets from noisy data 
only) while providing stable supervised training.

Usage:
    python train_iddm_torch.py
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import time

# Import data preparation and SVD
from noise_semi_clean import (
    CONFIG,
    load_and_scale_signals,
    add_gaussian_noise,
    apply_stft,
    split_data,
    stack_real_imag,
    generate_semi_clean_svd,
)

# Import PyTorch IDDM model and losses
from iddm_model_torch import create_iddm_model
from iddm_losses_torch import charbonnier_loss

# ISTFT and SNR functions (inline to avoid TensorFlow dependency from iddm_inference)
from scipy.signal import istft


def apply_istft_batch(stft_coefficients, window='hann'):
    """Apply Inverse STFT to batch of signals."""
    signals = []
    for stft_coeff in stft_coefficients:
        _, signal = istft(stft_coeff, window=window)
        signals.append(signal)
    return np.array(signals)


def calculate_snr(clean_signal, noisy_signal):
    """Calculate SNR between clean and noisy/denoised signal."""
    min_len = min(len(clean_signal), len(noisy_signal))
    clean = clean_signal[:min_len]
    noisy = noisy_signal[:min_len]
    
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((noisy - clean) ** 2)
    
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    return float('inf')


# =============================================================================
# Configuration
# =============================================================================
TRAIN_CONFIG = {
    # Training parameters
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-4,
    'patience': 10,
    
    # Inference - use 1 step for higher SNR, more steps for lower SNR
    'inference_steps': 3,
    
    # SVD parameters
    'energy_threshold': 0.70,
    
    # Checkpoint - include SNR in name to avoid overwriting
    'checkpoint_dir': 'checkpoints/iddm_torch',
}


# =============================================================================
# Data Preparation
# =============================================================================
def prepare_data_with_svd():
    """Prepare data with SVD semi-clean targets."""
    print("=" * 60)
    print("Preparing Data with SVD Semi-Clean Targets")
    print("=" * 60)
    
    # Load and scale
    print("\n[1] Loading and scaling signals...")
    scaled_signals, scaler = load_and_scale_signals(CONFIG['data_path'])
    
    # Add noise
    print("\n[2] Adding Gaussian noise...")
    noisy_signals = add_gaussian_noise(scaled_signals, CONFIG['target_snr_db'])
    noise_std = np.std(noisy_signals - scaled_signals)
    print(f"    Noise std: {noise_std:.4f}")
    
    # STFT
    print("\n[3] Applying STFT...")
    clean_stft = apply_stft(scaled_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])
    del scaled_signals
    gc.collect()
    
    noisy_stft = apply_stft(noisy_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])
    del noisy_signals
    gc.collect()
    
    # Split
    print("\n[4] Splitting data...")
    splits = split_data(
        noisy_stft, clean_stft,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )
    
    del noisy_stft, clean_stft
    gc.collect()
    
    # Generate SVD semi-clean targets
    print("\n[5] Generating SVD semi-clean targets (no ground truth needed)...")
    semi_clean_train = generate_semi_clean_svd(splits['X_train'], TRAIN_CONFIG['energy_threshold'])
    semi_clean_val = generate_semi_clean_svd(splits['X_val'], TRAIN_CONFIG['energy_threshold'])
    
    # Stack real/imag
    print("\n[6] Stacking real/imag components...")
    train_noisy = stack_real_imag(splits['X_train']).astype(np.float32)
    train_target = stack_real_imag(semi_clean_train).astype(np.float32)
    val_noisy = stack_real_imag(splits['X_val']).astype(np.float32)
    val_target = stack_real_imag(semi_clean_val).astype(np.float32)
    test_noisy = stack_real_imag(splits['X_test']).astype(np.float32)
    test_clean = stack_real_imag(splits['y_test']).astype(np.float32)
    
    print(f"    Train: {train_noisy.shape} -> {train_target.shape}")
    print(f"    Val:   {val_noisy.shape} -> {val_target.shape}")
    print(f"    Test:  {test_noisy.shape}")
    
    return {
        'train_noisy': train_noisy,
        'train_target': train_target,
        'val_noisy': val_noisy,
        'val_target': val_target,
        'test_noisy': test_noisy,
        'test_clean': test_clean,
    }


# =============================================================================
# Training Loop
# =============================================================================
def train_supervised(model, optimizer, data, config, device):
    """Train model with supervised learning."""
    print("\n" + "=" * 60)
    print("SUPERVISED TRAINING (PyTorch)")
    print("=" * 60)
    
    # Convert data to channels-first format: (N, H, W, C) -> (N, C, H, W)
    train_noisy = np.transpose(data['train_noisy'], (0, 3, 1, 2))
    train_target = np.transpose(data['train_target'], (0, 3, 1, 2))
    val_noisy = np.transpose(data['val_noisy'], (0, 3, 1, 2))
    val_target = np.transpose(data['val_target'], (0, 3, 1, 2))
    
    batch_size = config['batch_size']
    epochs = config['epochs']
    patience = config['patience']
    
    n_train = len(train_noisy)
    n_batches = n_train // batch_size
    
    print(f"Training samples: {n_train}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Early stopping patience: {patience}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        
        # Shuffle training data
        indices = np.random.permutation(n_train)
        train_losses = []
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            idx = indices[start:end]
            
            X_batch = torch.tensor(train_noisy[idx], dtype=torch.float32, device=device)
            y_batch = torch.tensor(train_target[idx], dtype=torch.float32, device=device)
            t_batch = torch.ones(len(idx), dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            
            # Forward pass - model expects (X_t, X_t1, t)
            output = model(X_batch, X_batch, t_batch)
            loss = charbonnier_loss(output, y_batch, epsilon=0.03)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for start in range(0, len(val_noisy), batch_size):
                end = min(start + batch_size, len(val_noisy))
                
                X_batch = torch.tensor(val_noisy[start:end], dtype=torch.float32, device=device)
                y_batch = torch.tensor(val_target[start:end], dtype=torch.float32, device=device)
                t_batch = torch.ones(end - start, dtype=torch.float32, device=device)
                
                output = model(X_batch, X_batch, t_batch)
                loss = charbonnier_loss(output, y_batch, epsilon=0.03)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(model, config, "best")
            marker = " *"
        else:
            patience_counter += 1
            marker = ""
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} "
              f"({epoch_time:.1f}s){marker}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    best_path = os.path.join(config['checkpoint_dir'], "best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print(f"Loaded best model (val_loss: {best_val_loss:.6f})")
    
    return history


# =============================================================================
# Inference
# =============================================================================
def run_inference(model, test_noisy, num_steps, batch_size, device, alpha=1.0):
    """Run iterative refinement with few steps."""
    print(f"\nRunning inference ({num_steps} refinement steps)...")
    
    model.eval()
    
    # Convert to channels-first: (N, H, W, C) -> (N, C, H, W)
    X_current = np.transpose(test_noisy.copy(), (0, 3, 1, 2))
    
    with torch.no_grad():
        for step in range(num_steps):
            X_next = np.zeros_like(X_current)
            
            for start in range(0, len(test_noisy), batch_size):
                end = min(start + batch_size, len(test_noisy))
                
                X_batch = torch.tensor(X_current[start:end], dtype=torch.float32, device=device)
                t_batch = torch.ones(end - start, dtype=torch.float32, device=device)
                
                output = model(X_batch, X_batch, t_batch)
                pred = output.cpu().numpy()
                
                # Blend with alpha
                X_next[start:end] = alpha * pred + (1 - alpha) * X_current[start:end]
            
            X_current = X_next
            print(f"  Refinement step {step+1}/{num_steps}")
    
    # Convert back to channels-last: (N, C, H, W) -> (N, H, W, C)
    return np.transpose(X_current, (0, 2, 3, 1))


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(denoised, clean, noisy):
    """Calculate SNR improvement."""
    # Convert to complex
    denoised_complex = denoised[..., 0] + 1j * denoised[..., 1]
    clean_complex = clean[..., 0] + 1j * clean[..., 1]
    noisy_complex = noisy[..., 0] + 1j * noisy[..., 1]
    
    # ISTFT
    denoised_signals = apply_istft_batch(denoised_complex)
    clean_signals = apply_istft_batch(clean_complex)
    noisy_signals = apply_istft_batch(noisy_complex)
    
    # SNR
    snr_denoised = []
    snr_noisy = []
    
    for i in range(len(clean_signals)):
        snr_denoised.append(calculate_snr(clean_signals[i], denoised_signals[i]))
        snr_noisy.append(calculate_snr(clean_signals[i], noisy_signals[i]))
    
    return {
        'denoised_avg': np.mean(snr_denoised),
        'denoised_std': np.std(snr_denoised),
        'noisy_avg': np.mean(snr_noisy),
        'improvement': np.mean(snr_denoised) - np.mean(snr_noisy),
    }


# =============================================================================
# Utilities
# =============================================================================
def save_checkpoint(model, config, name):
    """Save model checkpoint."""
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    path = os.path.join(config['checkpoint_dir'], f"{name}.pth")
    torch.save(model.state_dict(), path)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("IDDM Benchmark Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Prepare data
    data = prepare_data_with_svd()
    
    # Create model
    print("\nCreating IDDM model...")
    model = create_iddm_model((78, 78, 2), device=device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
    
    # Train
    history = train_supervised(model, optimizer, data, TRAIN_CONFIG, device)
    
    # Final checkpoint
    save_checkpoint(model, TRAIN_CONFIG, "final")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    denoised = run_inference(
        model, 
        data['test_noisy'],
        TRAIN_CONFIG['inference_steps'],
        TRAIN_CONFIG['batch_size'],
        device
    )
    
    results = evaluate(denoised, data['test_clean'], data['test_noisy'])
    
    print(f"\nNoisy baseline SNR: {results['noisy_avg']:.4f} dB")
    print(f"Denoised SNR:       {results['denoised_avg']:.4f} dB (±{results['denoised_std']:.4f})")
    print(f"Improvement:        {results['improvement']:.4f} dB")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, results, history


if __name__ == "__main__":
    model, results, history = main()
