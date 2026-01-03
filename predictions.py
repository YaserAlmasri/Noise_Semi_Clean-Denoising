"""
Prediction Pipeline with ISTFT Reconstruction

This script:
1. Loads a saved model checkpoint
2. Loads test data (or runs the preprocessing pipeline)
3. Runs predictions to get denoised STFT coefficients
4. Applies ISTFT to reconstruct time-domain signals
5. Calculates SNR metrics

Usage:
    python predictions.py
"""

import numpy as np
from scipy.signal import istft
from scipy.io import loadmat
import torch
import os

# Import from our modules (PyTorch)
from model_torch import build_unet, predict_and_reconstruct, get_device
from noise_semi_clean import (
    CONFIG, 
    load_and_scale_signals, 
    add_gaussian_noise, 
    apply_stft, 
    split_data,
    stack_real_imag,
    generate_semi_clean_svd
)


# =============================================================================
# Configuration
# =============================================================================
PREDICTION_CONFIG = {
    'checkpoint_path': 'checkpoints/best_model.pth',
    'stft_window': 'hann',  # Must match training STFT window
}


# =============================================================================
# ISTFT Reconstruction
# =============================================================================
def apply_istft(stft_coefficients, window='hann'):
    """
    Apply Inverse STFT to reconstruct time-domain signals.
    
    Args:
        stft_coefficients: Complex STFT coefficients of shape (N, freq, time)
        window: Window function (must match STFT window)
        
    Returns:
        Reconstructed time-domain signals
    """
    reconstructed_signals = []
    
    for stft_coeff in stft_coefficients:
        _, reconstructed = istft(stft_coeff, window=window)
        reconstructed_signals.append(reconstructed)
    
    return np.array(reconstructed_signals)


def reconstruct_all_signals(decoded_complex, semi_clean_test, clean_stft_test, noisy_stft_test, window='hann'):
    """
    Reconstruct all signal variants from STFT coefficients.
    
    Args:
        decoded_complex: Model predictions (denoised STFT)
        semi_clean_test: SVD semi-clean STFT
        clean_stft_test: Original clean STFT
        noisy_stft_test: Noisy STFT
        window: STFT window type
        
    Returns:
        Dictionary with all reconstructed signals
    """
    print("Reconstructing time-domain signals via ISTFT...")
    
    reconstructed = {
        'model_denoised': apply_istft(decoded_complex, window),
        'svd_semi_clean': apply_istft(semi_clean_test, window),
        'clean': apply_istft(clean_stft_test, window),
        'noisy': apply_istft(noisy_stft_test, window),
    }
    
    print(f"  Model denoised shape: {reconstructed['model_denoised'].shape}")
    print(f"  SVD semi-clean shape: {reconstructed['svd_semi_clean'].shape}")
    print(f"  Clean shape: {reconstructed['clean'].shape}")
    print(f"  Noisy shape: {reconstructed['noisy'].shape}")
    
    return reconstructed


# =============================================================================
# SNR Calculation
# =============================================================================
def calculate_snr(clean_signals, denoised_signals):
    """
    Calculate Signal-to-Noise Ratio between clean and denoised signals.
    
    SNR = 10 * log10(power_signal / power_noise)
    where noise = denoised - clean
    
    Args:
        clean_signals: Ground truth clean signals
        denoised_signals: Reconstructed/denoised signals
        
    Returns:
        Dictionary with individual SNR values and average
    """
    snr_values = []
    
    # Ensure same length for comparison
    min_len = min(clean_signals.shape[1], denoised_signals.shape[1])
    
    for i in range(len(clean_signals)):
        clean = clean_signals[i, :min_len]
        denoised = denoised_signals[i, :min_len]
        
        # Calculate signal power
        power_signal = np.mean(np.square(clean))
        
        # Calculate noise power (difference between denoised and clean)
        power_noise = np.mean(np.square(denoised - clean))
        
        # Avoid division by zero
        if power_noise > 0:
            snr_db = 10 * np.log10(power_signal / power_noise)
        else:
            snr_db = np.inf
            
        snr_values.append(snr_db)
    
    return {
        'individual': snr_values,
        'average': np.mean(snr_values),
        'std': np.std(snr_values),
    }


def evaluate_all_methods(reconstructed):
    """
    Evaluate SNR for all denoising methods.
    
    Args:
        reconstructed: Dictionary with reconstructed signals
        
    Returns:
        Dictionary with SNR results for each method
    """
    print("\n" + "=" * 60)
    print("SNR Evaluation Results")
    print("=" * 60)
    
    results = {}
    clean = reconstructed['clean']
    
    # Model denoised vs clean
    snr_model = calculate_snr(clean, reconstructed['model_denoised'])
    results['model_denoised'] = snr_model
    print(f"\nU-Net Model Denoised:")
    print(f"  Average SNR: {snr_model['average']:.4f} dB (±{snr_model['std']:.4f})")
    
    # SVD semi-clean vs clean
    snr_svd = calculate_snr(clean, reconstructed['svd_semi_clean'])
    results['svd_semi_clean'] = snr_svd
    print(f"\nSVD Semi-Clean:")
    print(f"  Average SNR: {snr_svd['average']:.4f} dB (±{snr_svd['std']:.4f})")
    
    # Noisy vs clean (baseline)
    snr_noisy = calculate_snr(clean, reconstructed['noisy'])
    results['noisy_baseline'] = snr_noisy
    print(f"\nNoisy Baseline:")
    print(f"  Average SNR: {snr_noisy['average']:.4f} dB (±{snr_noisy['std']:.4f})")
    
    # Improvement over baseline
    improvement = snr_model['average'] - snr_noisy['average']
    print(f"\nImprovement over noisy baseline: {improvement:.4f} dB")
    
    return results


# =============================================================================
# Load Model and Run Predictions
# =============================================================================
def load_saved_model(checkpoint_path):
    """
    Load a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to the saved model
        
    Returns:
        Loaded PyTorch model and device
    """
    print(f"Loading model from: {checkpoint_path}")
    
    device = get_device()
    
    # Create model and load weights
    model = build_unet((78, 78, 2))
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print("Model loaded successfully!")
    return model, device


def prepare_test_data():
    """
    Prepare test data by running the preprocessing pipeline.
    
    Returns:
        Dictionary with test data
    """
    import gc
    
    print("=" * 60)
    print("Preparing Test Data")
    print("=" * 60)
    
    # Load and process data
    print("\n[1] Loading and scaling signals...")
    scaled_signals, scaler = load_and_scale_signals(CONFIG['data_path'])
    
    print("\n[2] Adding Gaussian noise...")
    noisy_signals = add_gaussian_noise(scaled_signals, CONFIG['target_snr_db'])
    
    print("\n[3] Applying STFT...")
    clean_stft = apply_stft(scaled_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])
    del scaled_signals
    gc.collect()
    
    noisy_stft = apply_stft(noisy_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])
    del noisy_signals
    gc.collect()
    
    print("\n[4] Splitting data...")
    splits = split_data(
        noisy_stft, clean_stft,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )
    
    del noisy_stft, clean_stft
    gc.collect()
    
    # Generate semi-clean for test set
    print("\n[5] Generating SVD semi-clean for test set...")
    S_clean_test = generate_semi_clean_svd(splits['X_test'], CONFIG['energy_threshold'])
    
    return {
        'X_test': splits['X_test'],
        'y_test': splits['y_test'],
        'S_clean_test': S_clean_test,
    }


def run_predictions(model, test_data, device):
    """
    Run model predictions on test data.
    
    Args:
        model: Loaded PyTorch model
        test_data: Dictionary with test data
        device: Torch device
        
    Returns:
        Complex STFT predictions
    """
    print("\n[6] Running model predictions...")
    
    # Stack real/imag for model input
    test_noisy_stacked = stack_real_imag(test_data['X_test'])
    
    # Predict
    decoded_complex = predict_and_reconstruct(model, test_noisy_stacked, device)
    
    return decoded_complex


# =============================================================================
# Main Prediction Pipeline
# =============================================================================
def main():
    """Run the complete prediction and evaluation pipeline."""
    
    print("=" * 60)
    print("Signal Denoising - Prediction Pipeline")
    print("=" * 60)
    
    # Check if checkpoint exists
    if not os.path.exists(PREDICTION_CONFIG['checkpoint_path']):
        print(f"\nError: Checkpoint not found at {PREDICTION_CONFIG['checkpoint_path']}")
        print("Please run noise_semi_clean.py first to train the model.")
        return None
    
    # Load saved model
    model, device = load_saved_model(PREDICTION_CONFIG['checkpoint_path'])
    
    # Prepare test data
    test_data = prepare_test_data()
    
    # Run predictions
    decoded_complex = run_predictions(model, test_data, device)
    
    # Reconstruct time-domain signals
    print("\n[7] Reconstructing time-domain signals...")
    reconstructed = reconstruct_all_signals(
        decoded_complex=decoded_complex,
        semi_clean_test=test_data['S_clean_test'],
        clean_stft_test=test_data['y_test'],
        noisy_stft_test=test_data['X_test'],
        window=PREDICTION_CONFIG['stft_window']
    )
    
    # Evaluate SNR
    print("\n[8] Evaluating denoising performance...")
    snr_results = evaluate_all_methods(reconstructed)
    
    print("\n" + "=" * 60)
    print("Prediction Pipeline Complete!")
    print("=" * 60)
    
    return {
        'model': model,
        'test_data': test_data,
        'decoded_complex': decoded_complex,
        'reconstructed': reconstructed,
        'snr_results': snr_results,
    }


if __name__ == "__main__":
    results = main()
