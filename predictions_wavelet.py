"""
Prediction Pipeline for Impulsive + Gaussian Mixed Noise Experiment
(WAVELET TRANSFORM Version)

This script:
1. Loads the saved wavelet model checkpoint (best_model_wavelet_{snr}dB.pth)
2. Runs the preprocessing pipeline with impulsive+Gaussian noise to get test data
3. Applies DWT and reshapes to square 2D matrices
4. Runs model predictions to get denoised wavelet coefficient matrices
5. Applies inverse DWT to reconstruct time-domain signals
6. Calculates SNR metrics for all methods
7. Plots the top 3 most-improved signals as 4-panel figures:
   [Clean | Noisy (Imp.+Gauss.) | SVD Semi-Clean | U-Net Denoised]
8. Plots STFT spectrograms for the same signals

Usage:
    python predictions_wavelet.py
"""

import gc
import numpy as np
import pywt
from scipy.signal import stft as scipy_stft
import torch
import os
import matplotlib.pyplot as plt

# Import from our modules
from model_torch import build_unet, get_device
from noise_semi_clean_wavelet import (
    CONFIG,
    load_and_scale_signals,
    add_impulsive_gaussian_noise,
    apply_dwt,
    apply_idwt,
    compute_dwt_metadata,
    split_data,
    stack_for_unet,
    generate_semi_clean_svd,
    predict_wavelet,
)


# =============================================================================
# Configuration
# =============================================================================
PREDICTION_CONFIG = {
    'top_k_plots': 3,               # Number of best-improved signals to plot
    'extra_signal_indices': [342],   # Additional test signal indices to always plot
    'snr_levels': [-5, 0, 5],       # SNR levels to evaluate
    'plot_snr_db': 0,               # Only generate plots for this SNR level
    # Checkpoint template — must match training script
    'checkpoint_template': 'checkpoints/best_model_wavelet_{snr}dB.pth',
}


# =============================================================================
# Inverse DWT Reconstruction
# =============================================================================
def reconstruct_all_signals(decoded_matrices, semi_clean_test, clean_dwt_test,
                            noisy_dwt_test, dwt_meta):
    """
    Reconstruct all signal variants from wavelet coefficient matrices
    using inverse DWT.

    Args:
        decoded_matrices: Model predictions (denoised 2D matrices)
        semi_clean_test: SVD semi-clean 2D matrices
        clean_dwt_test: Original clean 2D matrices
        noisy_dwt_test: Noisy 2D matrices
        dwt_meta: DWT metadata for inverse transform

    Returns:
        Dictionary with all reconstructed time-domain signals
    """
    print("Reconstructing time-domain signals via inverse DWT...")

    reconstructed = {
        'model_denoised': apply_idwt(decoded_matrices, dwt_meta),
        'svd_semi_clean': apply_idwt(semi_clean_test, dwt_meta),
        'clean': apply_idwt(clean_dwt_test, dwt_meta),
        'noisy': apply_idwt(noisy_dwt_test, dwt_meta),
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

        power_signal = np.mean(np.square(clean))
        power_noise = np.mean(np.square(denoised - clean))

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
    print("SNR Evaluation Results (Wavelet — Impulsive + Gaussian Noise)")
    print("=" * 60)

    results = {}
    clean = reconstructed['clean']

    snr_model = calculate_snr(clean, reconstructed['model_denoised'])
    results['model_denoised'] = snr_model
    print(f"\nU-Net Model Denoised:")
    print(f"  Average SNR: {snr_model['average']:.4f} dB (±{snr_model['std']:.4f})")

    snr_svd = calculate_snr(clean, reconstructed['svd_semi_clean'])
    results['svd_semi_clean'] = snr_svd
    print(f"\nSVD Semi-Clean:")
    print(f"  Average SNR: {snr_svd['average']:.4f} dB (±{snr_svd['std']:.4f})")

    snr_noisy = calculate_snr(clean, reconstructed['noisy'])
    results['noisy_baseline'] = snr_noisy
    print(f"\nNoisy Baseline (Impulsive + Gaussian):")
    print(f"  Average SNR: {snr_noisy['average']:.4f} dB (±{snr_noisy['std']:.4f})")

    improvement_model = snr_model['average'] - snr_noisy['average']
    improvement_svd = snr_svd['average'] - snr_noisy['average']
    print(f"\nImprovement over noisy baseline:")
    print(f"  U-Net Model:    +{improvement_model:.4f} dB")
    print(f"  SVD Semi-Clean: +{improvement_svd:.4f} dB")

    return results


# =============================================================================
# Load Model and Run Predictions
# =============================================================================
def load_saved_model(checkpoint_path, input_shape):
    """
    Load a trained U-Net model from checkpoint.

    Args:
        checkpoint_path: Path to the saved model
        input_shape: Input shape tuple (H, W, C) for building the model

    Returns:
        Loaded PyTorch model and device
    """
    print(f"Loading model from: {checkpoint_path}")

    device = get_device()

    model = build_unet(input_shape)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Model loaded successfully!")
    return model, device


def prepare_test_data(target_snr_db):
    """
    Prepare test data by running the preprocessing pipeline
    with impulsive + Gaussian mixed noise at a given SNR level,
    using wavelet transform.

    Args:
        target_snr_db: Target SNR in dB for noise generation

    Returns:
        Dictionary with test data and dwt_meta
    """
    print("=" * 60)
    print(f"Preparing Test Data (Wavelet — Imp.+Gauss. Noise, {target_snr_db} dB)")
    print("=" * 60)

    print("\n[1] Loading and scaling signals...")
    scaled_signals, scaler = load_and_scale_signals(CONFIG['data_path'])

    print(f"\n[2] Adding impulsive + Gaussian mixed noise at {target_snr_db} dB...")
    noisy_signals = add_impulsive_gaussian_noise(
        scaled_signals,
        target_snr_db,
        impulse_prob=CONFIG['impulse_probability'],
        noise_split_ratio=CONFIG['noise_split_ratio'],
    )

    print("\n[3] Applying DWT...")
    clean_dwt, dwt_meta = apply_dwt(
        scaled_signals, CONFIG['wavelet'], CONFIG['dwt_level']
    )
    del scaled_signals
    gc.collect()

    noisy_dwt, _ = apply_dwt(
        noisy_signals, CONFIG['wavelet'], CONFIG['dwt_level'], dwt_meta=dwt_meta
    )
    del noisy_signals
    gc.collect()

    print("\n[4] Splitting data...")
    splits = split_data(
        noisy_dwt, clean_dwt,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )

    del noisy_dwt, clean_dwt
    gc.collect()

    # Generate semi-clean for test set
    energy_threshold = CONFIG['energy_threshold'][target_snr_db]
    print(f"\n[5] Generating SVD semi-clean for test set (threshold={energy_threshold})...")
    S_clean_test = generate_semi_clean_svd(splits['X_test'], energy_threshold)

    return {
        'X_test': splits['X_test'],
        'y_test': splits['y_test'],
        'S_clean_test': S_clean_test,
        'dwt_meta': dwt_meta,
    }


def run_predictions(model, test_data, device):
    """
    Run model predictions on test data (wavelet domain).

    Args:
        model: Loaded PyTorch model
        test_data: Dictionary with test data
        device: Torch device

    Returns:
        Denoised 2D coefficient matrices
    """
    print("\n[6] Running model predictions...")

    test_noisy_stacked = stack_for_unet(test_data['X_test'])
    decoded_matrices = predict_wavelet(model, test_noisy_stacked, device)

    return decoded_matrices


# =============================================================================
# Helpers
# =============================================================================
def normalize_to_unit(signal):
    """
    Normalize a signal to the range [-1, 1] using peak-absolute scaling.

    Args:
        signal: 1-D numpy array

    Returns:
        Normalized signal (same shape)
    """
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal / peak


# =============================================================================
# Plotting — Top-K Best Improved Signals
# =============================================================================
def find_top_k_improved(reconstructed, snr_results, k=3):
    """
    Find the indices of the k signals with the largest SNR improvement
    from noisy baseline to U-Net denoised.

    Args:
        reconstructed: Dictionary with reconstructed signals
        snr_results: Dictionary with per-method SNR results
        k: Number of top signals to return

    Returns:
        List of (index, noisy_snr, model_snr, improvement) tuples
    """
    noisy_snrs = snr_results['noisy_baseline']['individual']
    model_snrs = snr_results['model_denoised']['individual']

    improvements = []
    for i in range(len(noisy_snrs)):
        imp = model_snrs[i] - noisy_snrs[i]
        improvements.append((i, noisy_snrs[i], model_snrs[i], imp))

    improvements.sort(key=lambda x: x[3], reverse=True)

    return improvements[:k]


def _plot_single_signal(idx, reconstructed, snr_results, min_len, normalize,
                        save_dir, label):
    """
    Plot a single signal as a 4-panel figure and save as PNG + EPS.

    IEEE journal style: titles 24 pt, axis labels 22 pt, tick labels 18 pt,
    linewidth 0.8, DPI 300, colored lines.

    Args:
        idx: Index of the signal in the test set
        reconstructed: Dictionary with reconstructed signals
        snr_results: Dictionary with per-method SNR results
        min_len: Common signal length
        normalize: If True, normalize amplitudes to [-1, 1]
        save_dir: Directory to save plots
        label: Filename label (e.g. 'rank1' or 'extra')
    """
    noisy_snr = snr_results['noisy_baseline']['individual'][idx]
    svd_snr   = snr_results['svd_semi_clean']['individual'][idx]
    model_snr = snr_results['model_denoised']['individual'][idx]
    improvement = model_snr - noisy_snr

    print(f"\n  [{label}] Test signal #{idx}")
    print(f"    Noisy SNR:      {noisy_snr:.2f} dB")
    print(f"    SVD SNR:        {svd_snr:.2f} dB")
    print(f"    U-Net SNR:      {model_snr:.2f} dB")
    print(f"    Improvement:    +{improvement:.2f} dB")

    clean_sig = reconstructed['clean'][idx, :min_len]
    noisy_sig = reconstructed['noisy'][idx, :min_len]
    svd_sig   = reconstructed['svd_semi_clean'][idx, :min_len]
    unet_sig  = reconstructed['model_denoised'][idx, :min_len]

    if normalize:
        clean_sig = normalize_to_unit(clean_sig)
        noisy_sig = normalize_to_unit(noisy_sig)
        svd_sig   = normalize_to_unit(svd_sig)
        unet_sig  = normalize_to_unit(unet_sig)

    x = np.arange(min_len)

    fig, axs = plt.subplots(1, 4, figsize=(25, 5))

    panels = [
        ('Relatively Clean Signal', clean_sig, '#2ecc71'),
        ('Noisy Signal',            noisy_sig, '#e74c3c'),
        ('Semi-clean Signal',       svd_sig,   '#3498db'),
        ('Reconstructed Clean Signal', unet_sig, '#9b59b6'),
    ]

    for ax, (title, sig, color) in zip(axs, panels):
        ax.plot(x, sig, color=color, linewidth=0.8)
        ax.set_xlabel('Sample No.', fontsize=22)
        ax.set_ylabel('Amplitude', fontsize=22)
        ax.set_title(title, fontsize=24)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)

    if normalize:
        for ax in axs:
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks(np.arange(-1, 1.25, 0.25))

    plt.tight_layout()

    png_path = os.path.join(save_dir, f'{label}_signal{idx}_improvement.png')
    eps_path = os.path.join(save_dir, f'{label}_signal{idx}_improvement.eps')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"    Saved: {png_path}")
    print(f"    Saved: {eps_path}")

    plt.show()
    plt.close(fig)


def plot_top_signals(reconstructed, snr_results, top_k=3, normalize=False,
                     extra_indices=None, save_dir='plots_wavelet'):
    """
    Plot the top-k most improved signals as 4-panel figures,
    plus any extra signal indices that are not already in the top-k.

    IEEE journal style: colored lines, titles 24 pt, labels 22 pt, ticks 18 pt.

    Args:
        reconstructed: Dictionary with reconstructed signals
        snr_results: Dictionary with per-method SNR results
        top_k: Number of top-improved signals to plot
        normalize: If True, normalize amplitudes to [-1, 1]
        extra_indices: List of additional test-set indices to always plot
        save_dir: Directory to save plots

    Returns:
        all_indices: List of all plotted test-set indices
    """
    os.makedirs(save_dir, exist_ok=True)
    if extra_indices is None:
        extra_indices = []

    top_signals = find_top_k_improved(reconstructed, snr_results, k=top_k)
    top_indices = {t[0] for t in top_signals}

    min_len = min(
        reconstructed['clean'].shape[1],
        reconstructed['noisy'].shape[1],
        reconstructed['svd_semi_clean'].shape[1],
        reconstructed['model_denoised'].shape[1],
    )

    svd_snrs = snr_results['svd_semi_clean']['individual']
    n_test = len(snr_results['noisy_baseline']['individual'])

    print(f"\n{'='*60}")
    print(f"Plotting Top {top_k} Most Improved Signals (Wavelet)")
    print(f"{'='*60}")

    all_indices = []
    for rank, (idx, _, _, _) in enumerate(top_signals, 1):
        _plot_single_signal(idx, reconstructed, snr_results, min_len,
                            normalize, save_dir, label=f'rank{rank}')
        all_indices.append(idx)

    for extra_idx in extra_indices:
        if extra_idx in top_indices:
            print(f"\n  Signal #{extra_idx} is already in top-{top_k}, skipping duplicate.")
            continue
        if extra_idx >= n_test:
            print(f"\n  Warning: Signal #{extra_idx} out of range (test set has {n_test} signals), skipping.")
            continue
        _plot_single_signal(extra_idx, reconstructed, snr_results, min_len,
                            normalize, save_dir, label=f'extra_signal')
        all_indices.append(extra_idx)

    # --- Summary bar chart ---
    all_entries = []
    for idx in all_indices:
        noisy_snr = snr_results['noisy_baseline']['individual'][idx]
        model_snr = snr_results['model_denoised']['individual'][idx]
        all_entries.append((idx, noisy_snr, model_snr))

    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))

    indices_labels = [f"Sig #{e[0]}" for e in all_entries]
    noisy_vals = [e[1] for e in all_entries]
    svd_vals = [svd_snrs[e[0]] for e in all_entries]
    model_vals = [e[2] for e in all_entries]

    x = np.arange(len(all_entries))
    bar_width = 0.25

    bars_noisy = ax_bar.bar(x - bar_width, noisy_vals, bar_width,
                             label='Noisy (Imp.+Gauss.)', color='#e74c3c', alpha=0.85)
    bars_svd = ax_bar.bar(x, svd_vals, bar_width,
                           label='SVD Semi-Clean', color='#3498db', alpha=0.85)
    bars_model = ax_bar.bar(x + bar_width, model_vals, bar_width,
                             label='U-Net Denoised', color='#9b59b6', alpha=0.85)

    ax_bar.set_xlabel('Test Signal', fontsize=22)
    ax_bar.set_ylabel('SNR (dB)', fontsize=22)
    ax_bar.set_title('SNR Comparison \u2014 Wavelet (Imp.+Gauss. Noise)',
                      fontsize=24, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(indices_labels, fontsize=18)
    ax_bar.legend(fontsize=16)
    ax_bar.grid(True, axis='y')
    ax_bar.tick_params(axis='both', which='major', labelsize=18)

    for bars in [bars_noisy, bars_svd, bars_model]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    summary_png = os.path.join(save_dir, 'summary_snr_comparison.png')
    summary_eps = os.path.join(save_dir, 'summary_snr_comparison.eps')
    fig_bar.savefig(summary_png, dpi=300, bbox_inches='tight')
    fig_bar.savefig(summary_eps, format='eps', bbox_inches='tight')
    print(f"\n  Summary bar chart saved: {summary_png}")
    print(f"  Summary bar chart saved: {summary_eps}")
    plt.show()
    plt.close(fig_bar)

    return all_indices


def _plot_single_stft(idx, reconstructed, min_len, get_stft_magnitude,
                      save_dir, label):
    """
    Plot STFT spectrograms for a single signal as a 4-panel figure.

    IEEE journal style: titles 24 pt, axis labels 22 pt, ticks 18 pt,
    colorbar label 22 pt, colorbar ticks 16 pt, DPI 300.

    Args:
        idx: Index of the signal in the test set
        reconstructed: Dictionary with time-domain reconstructed signals
        min_len: Common signal length
        get_stft_magnitude: Function to compute STFT magnitude in dB
        save_dir: Directory to save plots
        label: Filename label
    """
    clean_sig = reconstructed['clean'][idx, :min_len]
    noisy_sig = reconstructed['noisy'][idx, :min_len]
    svd_sig   = reconstructed['svd_semi_clean'][idx, :min_len]
    unet_sig  = reconstructed['model_denoised'][idx, :min_len]

    frequencies, times, stft_clean = get_stft_magnitude(clean_sig)
    _, _, stft_noisy = get_stft_magnitude(noisy_sig)
    _, _, stft_svd   = get_stft_magnitude(svd_sig)
    _, _, stft_unet  = get_stft_magnitude(unet_sig)

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))

    panels = [
        ('Relatively Clean Spectrogram', stft_clean),
        ('Noisy Spectrogram',            stft_noisy),
        ('Semi-clean Spectrogram',       stft_svd),
        ('Denoised Spectrogram',         stft_unet),
    ]

    for ax, (title, stft_matrix) in zip(axes, panels):
        im = ax.pcolormesh(times, frequencies, stft_matrix,
                           vmin=-100, vmax=0)
        ax.set_title(title, fontsize=24)
        ax.set_xlabel('Time [s]', fontsize=22)
        ax.set_ylabel('Frequency [Hz]', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Magnitude [dB]', fontsize=22)
        cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()

    png_path = os.path.join(save_dir, f'{label}_signal{idx}_stft.png')
    eps_path = os.path.join(save_dir, f'{label}_signal{idx}_stft.eps')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"  [{label}] Test signal #{idx} STFT saved: {png_path}")
    print(f"  [{label}] Test signal #{idx} STFT saved: {eps_path}")

    plt.show()
    plt.close(fig)


def plot_top_signals_stft(all_indices, reconstructed, snr_results,
                          sampling_rate=200, nperseg=91,
                          save_dir='plots_wavelet'):
    """
    Plot STFT magnitude spectrograms for all plotted signal indices.

    Note: We compute STFT on the reconstructed TIME-DOMAIN signals (after
    inverse DWT). This lets us compare spectrograms across methods even
    though the U-Net operates in the wavelet domain.

    Args:
        all_indices: List of test-set indices to plot
        reconstructed: Dictionary with time-domain reconstructed signals
        snr_results: Dictionary with per-method SNR results
        sampling_rate: Sampling rate for STFT (default 200 Hz)
        nperseg: Segment length for STFT (default 91)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    epsilon = 1e-10

    min_len = min(
        reconstructed['clean'].shape[1],
        reconstructed['noisy'].shape[1],
        reconstructed['svd_semi_clean'].shape[1],
        reconstructed['model_denoised'].shape[1],
    )

    def get_stft_magnitude(signal):
        """Compute STFT magnitude in dB."""
        signal = np.abs(signal) + epsilon
        frequencies, times, magnitude = scipy_stft(
            signal, window='hann', fs=sampling_rate, nperseg=nperseg
        )
        magnitude = magnitude + epsilon
        stft_matrix = 20 * np.log10(np.abs(magnitude))
        return frequencies, times, stft_matrix

    print(f"\n{'='*60}")
    print(f"Plotting STFT Spectrograms for {len(all_indices)} Signals (Wavelet)")
    print(f"{'='*60}")

    top_signals = find_top_k_improved(reconstructed, snr_results, k=len(all_indices))
    top_rank_map = {t[0]: rank for rank, t in enumerate(top_signals, 1)}

    for idx in all_indices:
        if idx in top_rank_map and top_rank_map[idx] <= 3:
            label = f'rank{top_rank_map[idx]}'
        else:
            label = 'extra_signal'
        _plot_single_stft(idx, reconstructed, min_len, get_stft_magnitude,
                          save_dir, label)


# =============================================================================
# Plotting — CWT Scalogram
# =============================================================================
def _plot_single_scalogram(idx, reconstructed, min_len, scales, wavelet_name,
                           sampling_rate, save_dir, label):
    """
    Plot CWT scalogram for a single signal as a 4-panel figure.

    IEEE journal style: titles 24 pt, axis labels 22 pt, ticks 18 pt,
    colorbar label 22 pt, colorbar ticks 16 pt, DPI 300.

    Args:
        idx: Index of the signal in the test set
        reconstructed: Dictionary with time-domain reconstructed signals
        min_len: Common signal length
        scales: Array of CWT scales
        wavelet_name: CWT wavelet name (e.g. 'cmor1.5-1.0')
        sampling_rate: Sampling rate in Hz
        save_dir: Directory to save plots
        label: Filename label
    """
    epsilon = 1e-10

    clean_sig = reconstructed['clean'][idx, :min_len]
    noisy_sig = reconstructed['noisy'][idx, :min_len]
    svd_sig   = reconstructed['svd_semi_clean'][idx, :min_len]
    unet_sig  = reconstructed['model_denoised'][idx, :min_len]

    time_axis = np.arange(min_len) / sampling_rate

    # Compute CWT for each variant
    def cwt_magnitude_db(signal):
        coeffs, freqs = pywt.cwt(signal, scales, wavelet_name,
                                 sampling_period=1.0 / sampling_rate)
        mag_db = 20 * np.log10(np.abs(coeffs) + epsilon)
        return freqs, mag_db

    freqs, cwt_clean = cwt_magnitude_db(clean_sig)
    _,     cwt_noisy = cwt_magnitude_db(noisy_sig)
    _,     cwt_svd   = cwt_magnitude_db(svd_sig)
    _,     cwt_unet  = cwt_magnitude_db(unet_sig)

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))

    panels = [
        ('Relatively Clean Scalogram', cwt_clean),
        ('Noisy Scalogram',            cwt_noisy),
        ('Semi-clean Scalogram',       cwt_svd),
        ('Denoised Scalogram',         cwt_unet),
    ]

    for ax, (title, cwt_matrix) in zip(axes, panels):
        im = ax.pcolormesh(time_axis, freqs, cwt_matrix, vmin=-100, vmax=0)
        ax.set_title(title, fontsize=24)
        ax.set_xlabel('Time [s]', fontsize=22)
        ax.set_ylabel('Frequency [Hz]', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Magnitude [dB]', fontsize=22)
        cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()

    png_path = os.path.join(save_dir, f'{label}_signal{idx}_scalogram.png')
    eps_path = os.path.join(save_dir, f'{label}_signal{idx}_scalogram.eps')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"  [{label}] Test signal #{idx} scalogram saved: {png_path}")
    print(f"  [{label}] Test signal #{idx} scalogram saved: {eps_path}")

    plt.show()
    plt.close(fig)


def plot_top_signals_scalogram(all_indices, reconstructed, snr_results,
                               sampling_rate=200, n_scales=128,
                               cwt_wavelet='cmor1.5-1.0',
                               save_dir='plots_wavelet'):
    """
    Plot CWT scalograms for all plotted signal indices.

    The scalogram is the wavelet-domain equivalent of the STFT spectrogram.
    Uses the Continuous Wavelet Transform (CWT) with a complex Morlet
    wavelet to produce a scale-vs-time magnitude plot.

    Args:
        all_indices: List of test-set indices to plot
        reconstructed: Dictionary with time-domain reconstructed signals
        snr_results: Dictionary with per-method SNR results
        sampling_rate: Sampling rate in Hz (default 200)
        n_scales: Number of CWT scales (default 128)
        cwt_wavelet: Wavelet for CWT (default 'cmor1.5-1.0')
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    min_len = min(
        reconstructed['clean'].shape[1],
        reconstructed['noisy'].shape[1],
        reconstructed['svd_semi_clean'].shape[1],
        reconstructed['model_denoised'].shape[1],
    )

    # Logarithmically-spaced scales covering useful frequency range
    max_freq = sampling_rate / 2
    min_freq = sampling_rate / min_len  # lowest resolvable frequency
    scales = np.logspace(
        np.log10(1), np.log10(min_len / 2), num=n_scales
    )

    print(f"\n{'='*60}")
    print(f"Plotting CWT Scalograms for {len(all_indices)} Signals (Wavelet)")
    print(f"  CWT wavelet: {cwt_wavelet}, scales: {n_scales}")
    print(f"{'='*60}")

    top_signals = find_top_k_improved(reconstructed, snr_results, k=len(all_indices))
    top_rank_map = {t[0]: rank for rank, t in enumerate(top_signals, 1)}

    for idx in all_indices:
        if idx in top_rank_map and top_rank_map[idx] <= 3:
            label = f'rank{top_rank_map[idx]}'
        else:
            label = 'extra_signal'
        _plot_single_scalogram(idx, reconstructed, min_len, scales,
                               cwt_wavelet, sampling_rate, save_dir, label)


# =============================================================================
# Main Prediction Pipeline
# =============================================================================
def run_single_snr_evaluation(target_snr_db, do_plot=False):
    """
    Evaluate a trained wavelet model for one SNR level.

    Args:
        target_snr_db: The SNR level in dB
        do_plot: If True, generate time-domain + STFT plots

    Returns:
        Dictionary with SNR results and reconstructed signals
    """
    checkpoint_path = PREDICTION_CONFIG['checkpoint_template'].format(snr=target_snr_db)

    print("\n" + "#" * 60)
    print(f"  Evaluating SNR = {target_snr_db} dB (Wavelet)")
    print(f"  Checkpoint: {checkpoint_path}")
    print("#" * 60)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: Checkpoint not found at {checkpoint_path}")
        print("  Please run noise_semi_clean_wavelet.py first.")
        return None

    # Prepare test data at this SNR level (need dwt_meta for input_shape)
    test_data = prepare_test_data(target_snr_db)
    dwt_meta = test_data['dwt_meta']
    side = dwt_meta['square_side']
    input_shape = (side, side, 1)

    # Load saved model with correct input_shape
    model, device = load_saved_model(checkpoint_path, input_shape)

    # Run predictions
    decoded_matrices = run_predictions(model, test_data, device)

    # Reconstruct time-domain signals via inverse DWT
    print("\n[7] Reconstructing time-domain signals via inverse DWT...")
    reconstructed = reconstruct_all_signals(
        decoded_matrices=decoded_matrices,
        semi_clean_test=test_data['S_clean_test'],
        clean_dwt_test=test_data['y_test'],
        noisy_dwt_test=test_data['X_test'],
        dwt_meta=dwt_meta,
    )

    # Evaluate SNR
    print("\n[8] Evaluating denoising performance...")
    snr_results = evaluate_all_methods(reconstructed)

    # Plots only for the selected SNR level
    if do_plot:
        print("\n[9] Plotting best-improved signals (time-domain)...")
        all_indices = plot_top_signals(
            reconstructed,
            snr_results,
            top_k=PREDICTION_CONFIG['top_k_plots'],
            extra_indices=PREDICTION_CONFIG.get('extra_signal_indices', []),
        )

        print("\n[10] Plotting STFT spectrograms...")
        plot_top_signals_stft(
            all_indices=all_indices,
            reconstructed=reconstructed,
            snr_results=snr_results,
        )

        print("\n[11] Plotting CWT scalograms...")
        plot_top_signals_scalogram(
            all_indices=all_indices,
            reconstructed=reconstructed,
            snr_results=snr_results,
        )

    return {
        'target_snr_db': target_snr_db,
        'snr_results': snr_results,
        'reconstructed': reconstructed,
    }


def main():
    """Run evaluation for all SNR levels, print summary table."""

    print("=" * 60)
    print("Signal Denoising — Prediction Pipeline (Wavelet, Multi-SNR)")
    print("  ** Impulsive + Gaussian Mixed Noise Experiment **")
    print(f"  SNR levels: {PREDICTION_CONFIG['snr_levels']}")
    print(f"  Plots for: {PREDICTION_CONFIG['plot_snr_db']} dB only")
    print("=" * 60)

    all_results = {}
    for snr_db in PREDICTION_CONFIG['snr_levels']:
        do_plot = (snr_db == PREDICTION_CONFIG['plot_snr_db'])
        result = run_single_snr_evaluation(snr_db, do_plot=do_plot)
        if result is not None:
            all_results[snr_db] = result

    # ---- Summary table across all SNR levels ----
    print("\n" + "=" * 80)
    print("SUMMARY — Average SNR Improvement (Wavelet — Imp.+Gauss. Noise)")
    print("=" * 80)
    print(f"{'Target SNR':>12s} | {'Noisy (dB)':>12s} | {'SVD (dB)':>12s} | "
          f"{'U-Net (dB)':>12s} | {'Improvement':>12s}")
    print("-" * 80)

    for snr_db in sorted(all_results.keys()):
        r = all_results[snr_db]['snr_results']
        noisy_avg = r['noisy_baseline']['average']
        svd_avg   = r['svd_semi_clean']['average']
        model_avg = r['model_denoised']['average']
        improvement = model_avg - noisy_avg
        print(f"{snr_db:>+10d} dB | {noisy_avg:>12.4f} | {svd_avg:>12.4f} | "
              f"{model_avg:>12.4f} | {improvement:>+12.4f}")

    print("=" * 80)
    print("\nPrediction Pipeline Complete!")

    return all_results


if __name__ == "__main__":
    results = main()
