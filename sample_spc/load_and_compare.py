#!/usr/bin/env python3
import os

os.environ["TTTRLIB_VERBOSE"] = "1"  # Enable C++ verbose logging
"""
Load BH SPC file and compare reconstructed CLSM image against SPCM reference.

Uses the new BH-specific reading routines:
- parse_bh_set_file: Parse .set file to get image dimensions and pixel clock settings
- detect_frame1_extra_line: Detect BH Frame 1 initialization line anomaly
- get_adjusted_frame_markers: Get corrected frame markers that skip the anomaly
- CLSMImage with use_pixel_markers=True: Pixel marker-based binning for accurate reconstruction
"""

import numpy as np
import tifffile
import tttrlib
import sys
from tttrlib import CLSM_BH_SPC130

# Enable verbose debugging
print("=" * 60)
print("VERBOSE MODE ENABLED")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"tttrlib location: {tttrlib.__file__}")
print()


def compute_rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Root Mean Square Error between two images."""
    return float(
        np.sqrt(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    )


def main():
    # File paths
    sample_dir = os.path.dirname(os.path.abspath(__file__))
    spc_file = os.path.join(sample_dir, "01_HEKn_before_720nm_63xWI_10dbm_m1.spc")
    reference_tif = os.path.join(
        sample_dir,
        "01_HEKn_before_720nm_63xWI_10dbm_m1_spcm_reference_intensity_image.tif",
    )

    # =========================================================================
    # Step 1: Load reference TIFF image
    # =========================================================================
    print(f"Loading reference TIFF: {reference_tif}")
    reference_img = tifffile.imread(reference_tif)
    print(f"Reference image shape: {reference_img.shape}")
    print(f"Reference image dtype: {reference_img.dtype}")
    print(f"Reference total photons: {reference_img.sum()}")

    # =========================================================================
    # Step 2: Load SPC file using TTTR
    # =========================================================================
    print(f"\nLoading SPC file: {spc_file}")
    # C++ automatically finds and parses the .set file if it exists in same folder
    tttr = tttrlib.TTTR(spc_file, "SPC-130")
    print(f"Loaded {tttr.n_valid_events} events")

    # =========================================================================
    # Step 3: Create CLSMImage with BH reading routine
    # =========================================================================
    print("\nBuilding CLSMImage with BH reading routine...")
    # All defaults (marker channels, dimensions, Frame 1 adjustment)
    # are handled automatically by the CLSM_BH_SPC130 reading routine.
    clsm_params = {
        "reading_routine": CLSM_BH_SPC130,
    }

    clsm = tttrlib.CLSMImage(tttr_data=tttr, **clsm_params)
    print("CLSMImage created successfully!")

    clsm.fill(tttr_data=tttr, channels=[0])

    # Get intensity image
    clsm_intensity = clsm.intensity
    print(
        f"\nCLSM image shape: {clsm_intensity.shape}"
    )  # (n_frames, n_lines, n_pixels)

    # =========================================================================
    # Step 4: Sum frames and prepare for comparison
    # =========================================================================
    # Sum all frames to get total intensity image
    clsm_summed = clsm_intensity.sum(axis=0)
    print(f"After summing frames: {clsm_summed.shape}")
    print(f"CLSM total photons: {clsm_summed.sum()}")

    # Handle shape mismatch if needed
    # Reference might have different dimensions; crop to match
    ref_shape = reference_img.shape
    if len(ref_shape) == 3:
        # Reference is multi-frame, sum it
        reference_summed = reference_img.sum(axis=0)
    else:
        reference_summed = reference_img

    print(f"Reference shape for comparison: {reference_summed.shape}")

    # Crop CLSM to match reference if needed
    min_lines = min(clsm_summed.shape[0], reference_summed.shape[0])
    min_pixels = min(clsm_summed.shape[1], reference_summed.shape[1])

    clsm_cropped = clsm_summed[:min_lines, :min_pixels]
    reference_cropped = reference_summed[:min_lines, :min_pixels]

    print(f"Comparison region: {clsm_cropped.shape}")

    # =========================================================================
    # Step 5: Compute RMSE
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    rmse = compute_rmse(clsm_cropped, reference_cropped)
    print(f"\nRMSE (CLSM vs Reference): {rmse:.6f}")

    # Additional statistics
    diff = clsm_cropped.astype(np.float64) - reference_cropped.astype(np.float64)
    print(f"\nDifference statistics:")
    print(f"  Min difference:  {diff.min():.2f}")
    print(f"  Max difference:  {diff.max():.2f}")
    print(f"  Mean difference: {diff.mean():.2f}")
    print(f"  Std difference:  {diff.std():.2f}")

    # Percentage of pixels with exact match
    exact_match_pct = (np.abs(diff) < 0.5).sum() / diff.size * 100
    print(f"\nPixels with exact match: {exact_match_pct:.2f}%")

    # Photon count comparison
    clsm_total = clsm_cropped.sum()
    ref_total = reference_cropped.sum()
    photon_diff = clsm_total - ref_total
    photon_diff_pct = photon_diff / ref_total * 100 if ref_total > 0 else 0
    print(f"\nPhoton count comparison:")
    print(f"  CLSM photons:      {clsm_total}")
    print(f"  Reference photons: {ref_total}")
    print(f"  Difference:        {photon_diff:+.0f} ({photon_diff_pct:+.2f}%)")

    print("\n" + "=" * 60)

    return rmse


if __name__ == "__main__":
    rmse = main()
