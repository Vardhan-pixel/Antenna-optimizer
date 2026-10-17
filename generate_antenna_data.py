"""
Synthetic Antenna Dataset Generator
------------------------------------
Generates a realistic dataset for a rectangular microstrip patch antenna
using standard design equations (Balanis' Antenna Theory).

Inputs sampled:  fr (frequency, GHz), er (dielectric constant), h (substrate height, mm)
Outputs computed: W (width, mm), L (length, mm), gain (dBi, approx),
                  bandwidth (%, approx), S11_min (dB, approx return loss at resonance)

Controlled noise is added to simulate real-world measurement/manufacturing variation.

Note on S11: True S11 requires full-wave EM simulation (HFSS/CST/MATLAB antenna toolbox).
Here we use a physics-inspired approximation: the minimum return loss (dip depth) is modeled
as a function of impedance matching quality (randomized within realistic bounds), and the full
S11-vs-frequency curve is generated using a resonance (Lorentzian-style notch) model centered
at fr with width controlled by the bandwidth. This is a common, legitimate way to approximate
S11 behavior for ML training purposes when EM simulation data isn't available -- worth stating
explicitly as an assumption in your report.
"""

import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
N_SAMPLES = 2000          # number of synthetic rows to generate
NOISE_LEVEL = 0.03        # 3% random noise on outputs
RANDOM_SEED = 42

# Realistic ranges for a student-level patch antenna project
FREQ_RANGE_GHZ = (1.0, 10.0)          # operating frequency
DIELECTRIC_OPTIONS = [2.2, 2.33, 3.38, 4.4, 6.15, 10.2]  # common substrates (RT/Duroid, FR4, RO4003, etc.)
HEIGHT_RANGE_MM = (0.5, 3.2)          # substrate thickness

C = 3e8  # speed of light, m/s

np.random.seed(RANDOM_SEED)


def add_noise(value, noise_level=NOISE_LEVEL):
    """Add gaussian noise proportional to the value."""
    noise = np.random.normal(0, noise_level * np.abs(value))
    return value + noise


def design_patch_antenna(fr_ghz, er, h_mm):
    """
    Compute patch antenna dimensions using standard transmission line model equations.
    fr_ghz : resonant frequency in GHz
    er     : dielectric constant of substrate
    h_mm   : substrate height in mm

    Returns: W (mm), L (mm), eff_dielectric, gain_est (dBi), bandwidth_est (%)
    """
    fr = fr_ghz * 1e9      # Hz
    h = h_mm * 1e-3        # meters

    # --- Width ---
    W = (C / (2 * fr)) * np.sqrt(2 / (er + 1))

    # --- Effective dielectric constant ---
    eff_er = ((er + 1) / 2) + ((er - 1) / 2) * (1 + 12 * h / W) ** -0.5

    # --- Length extension due to fringing ---
    num = (eff_er + 0.3) * (W / h + 0.264)
    den = (eff_er - 0.258) * (W / h + 0.8)
    delta_L = 0.412 * h * (num / den)

    # --- Actual patch length ---
    L = (C / (2 * fr * np.sqrt(eff_er))) - 2 * delta_L

    # Convert back to mm
    W_mm = W * 1e3
    L_mm = L * 1e3

    # --- Approximate gain estimate (empirical relation, dBi) ---
    # Larger patch (electrically) + higher effective dielectric -> higher directivity trend
    gain_est = 6.0 + 10 * np.log10((W_mm * L_mm) / (fr_ghz ** 2)) * 0.15
    gain_est = np.clip(gain_est, 2.0, 12.0)  # realistic patch antenna gain bounds

    # --- Approximate bandwidth estimate (%) using standard patch BW approximation ---
    # BW% ~ proportional to h/lambda0 and inversely to er (thinner/high-er substrates -> narrower BW)
    lambda0_mm = (C / fr) * 1e3
    bandwidth_est = 3.77 * ((er - 1) / er ** 2) * (h_mm / lambda0_mm) * (W_mm / L_mm) * 100
    bandwidth_est = np.clip(bandwidth_est, 0.5, 8.0)  # typical patch antenna BW range

    # --- Approximate S11 minimum (return loss depth at resonance, dB) ---
    # Well-matched antennas typically show S11 between -35 dB (excellent) and -10 dB (marginal).
    # Match quality is modeled as a function of the actual design parameters (not pure randomness),
    # so the relationship is learnable: thicker substrates relative to wavelength and moderate
    # dielectric constants generally allow better impedance matching in practice.
    h_over_lambda = h_mm / lambda0_mm  # electrical thickness, normalized to wavelength
    er_factor = 1.0 / np.sqrt(er)      # lower er tends to allow easier/better matching

    # Combine into a 0-1 "matchability" score using a smooth saturating function
    raw_score = 6.0 * h_over_lambda + 0.8 * er_factor
    match_quality = 1 / (1 + np.exp(-8 * (raw_score - 0.35)))  # sigmoid, centered near typical designs

    s11_min_db = -(9 + 24 * match_quality)
    s11_min_db = add_noise(s11_min_db, noise_level=0.04)  # small measurement-style noise only
    s11_min_db = np.clip(s11_min_db, -35.0, -8.0)

    return W_mm, L_mm, eff_er, gain_est, bandwidth_est, s11_min_db


def generate_s11_curve(fr_ghz, bandwidth_percent, s11_min_db, n_points=201, span_factor=3.0):
    """
    Generate a full S11-vs-frequency sweep for plotting, using a Lorentzian-style notch
    centered at the resonant frequency. The notch width is derived from the bandwidth.

    fr_ghz            : resonant (center) frequency, GHz
    bandwidth_percent : impedance bandwidth, %
    s11_min_db        : depth of the S11 dip at resonance, dB
    n_points          : number of frequency points in the sweep
    span_factor       : how many bandwidths wide the plotted sweep should span

    Returns: freqs (GHz array), s11_db (dB array)
    """
    bw_ghz = fr_ghz * (bandwidth_percent / 100.0)
    half_span = max(bw_ghz * span_factor, fr_ghz * 0.05)  # ensure a sensible minimum span

    freqs = np.linspace(fr_ghz - half_span, fr_ghz + half_span, n_points)
    freqs = np.clip(freqs, 0.05, None)  # avoid non-physical negative frequencies

    # Lorentzian notch: shallow far from resonance (~ -1 to -3 dB), deep at resonance (s11_min_db)
    gamma = max(bw_ghz / 2.0, 1e-4)  # half-width at half depth, tied to bandwidth
    baseline_db = -1.5
    depth = baseline_db - s11_min_db
    notch = depth / (1 + ((freqs - fr_ghz) / gamma) ** 2)
    s11_db = baseline_db - notch

    return freqs, s11_db


def generate_dataset(n=N_SAMPLES):
    rows = []
    for _ in range(n):
        fr_ghz = np.random.uniform(*FREQ_RANGE_GHZ)
        er = np.random.choice(DIELECTRIC_OPTIONS)
        h_mm = np.random.uniform(*HEIGHT_RANGE_MM)

        W, L, eff_er, gain, bw, s11_min = design_patch_antenna(fr_ghz, er, h_mm)

        # Add realistic noise to simulate manufacturing/measurement variation
        W_noisy = add_noise(W)
        L_noisy = add_noise(L)
        gain_noisy = add_noise(gain, noise_level=0.05)
        bw_noisy = add_noise(bw, noise_level=0.05)
        s11_noisy = add_noise(s11_min, noise_level=0.05)
        s11_noisy = np.clip(s11_noisy, -40.0, -5.0)

        rows.append({
            "frequency_GHz": round(fr_ghz, 3),
            "dielectric_constant": er,
            "substrate_height_mm": round(h_mm, 3),
            "effective_dielectric": round(eff_er, 4),
            "patch_width_mm": round(W_noisy, 4),
            "patch_length_mm": round(L_noisy, 4),
            "gain_dBi": round(gain_noisy, 3),
            "bandwidth_percent": round(bw_noisy, 3),
            "S11_min_dB": round(s11_noisy, 3),
        })

    return pd.DataFrame(rows)


def blend_with_real_data(synthetic_df, real_csv_path=None):
    """
    Optional: if you have real data (from your internship / MATLAB / measurements),
    load it here and concatenate it with the synthetic data, tagging the source.
    This lets your ML model learn from both physics-based and real samples.

    real_csv_path: path to your real data CSV. Expected columns should match
    (or be renamed to match) the synthetic dataset columns above.
    """
    synthetic_df["source"] = "synthetic"

    if real_csv_path:
        real_df = pd.read_csv(real_csv_path)
        real_df["source"] = "real"
        combined = pd.concat([synthetic_df, real_df], ignore_index=True)
        return combined

    return synthetic_df


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} synthetic antenna design samples...")
    df = generate_dataset(N_SAMPLES)

    # If you have real data, put its path here, e.g.:
    # df = blend_with_real_data(df, "my_real_antenna_data.csv")
    df = blend_with_real_data(df)

    output_path = "antenna_synthetic_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"Done. Saved {len(df)} rows to {output_path}")
    print("\nPreview:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
