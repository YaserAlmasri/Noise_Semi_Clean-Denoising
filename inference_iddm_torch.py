"""
IDDM Inference Script (PyTorch Version)

Load pre-trained weights and run inference on test data.

Usage:
    python inference_iddm_torch.py
    python inference_iddm_torch.py --checkpoint checkpoints/iddm_supervised_torch/best.pth
    python inference_iddm_torch.py --steps 1 --alpha 0.5
"""

import os
import gc
import argparse
import numpy as np
import torch

# Import data preparation
from noise_semi_clean import (
    CONFIG,
    load_and_scale_signals,
    add_gaussian_noise,
    apply_stft,
    split_data,
    stack_real_imag,
)

# Import PyTorch IDDM model
from iddm_model_torch import create_iddm_model

# ISTFT and SNR functions (inline to avoid TensorFlow dependency)
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


def load_test_data():
    """Load and prepare test data."""
    print("=" * 60)
    print("Loading Test Data")
    print("=" * 60)
    
    # Load and scale
    print("\n[1] Loading signals...")
    scaled_signals, scaler = load_and_scale_signals(CONFIG['data_path'])
    
    # Add noise
    print(f"[2] Adding noise at {CONFIG['target_snr_db']} dB SNR...")
    noisy_signals = add_gaussian_noise(scaled_signals, CONFIG['target_snr_db'])
    
    # STFT
    print("[3] Applying STFT...")
    clean_stft = apply_stft(scaled_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])
    noisy_stft = apply_stft(noisy_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])
    
    del scaled_signals, noisy_signals
    gc.collect()
    
    # Split - only need test set
    print("[4] Extracting test set...")
    splits = split_data(
        noisy_stft, clean_stft,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )
    
    del noisy_stft, clean_stft
    gc.collect()
    
    test_noisy = stack_real_imag(splits['X_test']).astype(np.float32)
    test_clean = stack_real_imag(splits['y_test']).astype(np.float32)
    
    print(f"    Test samples: {len(test_noisy)}")
    
    return test_noisy, test_clean


def run_inference(model, test_noisy, num_steps, batch_size, device, alpha=1.0):
    """
    Run inference with specified number of refinement steps.
    
    Args:
        model: Trained PyTorch model
        test_noisy: Noisy test data (N, H, W, C) - channels last
        num_steps: Number of refinement iterations
        batch_size: Batch size for inference
        device: Torch device
        alpha: Blending factor (0.0=no change, 0.5=half step, 1.0=full step)
    """
    alpha_str = f", alpha={alpha}" if alpha != 1.0 else ""
    print(f"\nRunning inference ({num_steps} step{'s' if num_steps > 1 else ''}{alpha_str})...")
    
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
                
                # Blend model output with input
                X_next[start:end] = alpha * pred + (1 - alpha) * X_current[start:end]
            
            X_current = X_next
            if num_steps > 1:
                print(f"  Step {step+1}/{num_steps} complete")
    
    print("  Inference complete!")
    
    # Convert back to channels-last: (N, C, H, W) -> (N, H, W, C)
    return np.transpose(X_current, (0, 2, 3, 1))


def evaluate(denoised, clean, noisy):
    """Calculate SNR metrics."""
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
        'denoised_signals': denoised_signals,
        'clean_signals': clean_signals,
        'noisy_signals': noisy_signals,
    }


def main():
    parser = argparse.ArgumentParser(description='IDDM Inference (PyTorch)')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/iddm_torch/best.pth',
                        help='Path to model weights')
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of refinement steps (1 for high SNR, 3-5 for low SNR)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Blending factor: 0.0=no change, 0.5=half step, 1.0=full step')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    args = parser.parse_args()
    
    print("=" * 60)
    print("IDDM Inference (PyTorch)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Inference steps: {args.steps}")
    print(f"Alpha (blend): {args.alpha}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    test_noisy, test_clean = load_test_data()
    
    # Create and load model
    print("\nLoading model...")
    model = create_iddm_model((78, 78, 2), device=device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded weights from: {args.checkpoint}")
    else:
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Run inference
    denoised = run_inference(model, test_noisy, args.steps, args.batch_size, device, args.alpha)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    results = evaluate(denoised, test_clean, test_noisy)
    
    print(f"\nNoisy baseline SNR: {results['noisy_avg']:.4f} dB")
    print(f"Denoised SNR:       {results['denoised_avg']:.4f} dB (±{results['denoised_std']:.4f})")
    print(f"Improvement:        {results['improvement']:.4f} dB")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
