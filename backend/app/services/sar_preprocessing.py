# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — Sentinel-1 SAR Preprocessing Pipeline
Performs radiometric calibration to sigma0 (dB), Lee speckle filtering,
and land-sea masking on raw SAR backscatter prior to ML segmentation.
"""

import numpy as np
from scipy.ndimage import uniform_filter, median_filter
from typing import Tuple, Optional

class SARPreprocessor:
    """
    Standard SAR preprocessor implementing:
    1. Radiometric calibration from raw DN to Sigma0 (dB).
    2. Adaptive Lee speckle filtering to suppress coherent microwave noise.
    3. Land/sea masking and backscatter normalization for neural network input.
    """

    @staticmethod
    def calibrate_to_sigma0_db(raw_amplitude: np.ndarray, calibration_factor: float = 1.0) -> np.ndarray:
        """
        Converts raw SAR amplitude / digital numbers to radiometric Sigma0 backscatter in decibels (dB).
        Formula: sigma0_db = 10 * log10(max(amplitude^2 * cal_factor, 1e-5))
        """
        intensity = np.square(np.maximum(raw_amplitude, 1e-4)) * calibration_factor
        sigma0_db = 10.0 * np.log10(np.maximum(intensity, 1e-5))
        # Clip to physical marine radar limits [-35 dB (calm dark slick) to +5 dB (steep land/vessel)]
        return np.clip(sigma0_db, -35.0, 5.0)

    @staticmethod
    def lee_speckle_filter(image: np.ndarray, window_size: int = 5, num_looks: int = 1) -> np.ndarray:
        """
        Adaptive Lee Filter for SAR multiplicative speckle noise reduction.
        Preserves sharp oil-slick damping edges while smoothing homogeneous water clutter.
        """
        # Ensure floating point array
        img = image.astype(np.float32)

        # Local mean
        local_mean = uniform_filter(img, size=window_size)
        # Local variance
        local_sqr_mean = uniform_filter(np.square(img), size=window_size)
        local_var = np.maximum(local_sqr_mean - np.square(local_mean), 1e-6)

        # Theoretical noise variance for multi-look SAR
        noise_var = 1.0 / max(num_looks, 1)

        # Weighting factor W = var / (var + noise_var * mean^2)
        weights = local_var / (local_var + noise_var * np.square(local_mean) + 1e-6)
        weights = np.clip(weights, 0.0, 1.0)

        # Filtered output: R = mean + W * (I - mean)
        filtered = local_mean + weights * (img - local_mean)
        return filtered

    @staticmethod
    def mask_land_pixels(
        sar_db: np.ndarray,
        land_threshold_db: float = -4.0,
        morph_cleanup: bool = True,
        bounds: Optional[dict] = None,
        transform: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a marine surface mask by excluding authoritative land terrain and nodata.
        Uses LandSeaMaskService (Natural Earth 50m + regional coastlines) when bounds are available,
        falling back to advanced radiometric + texture discrimination.
        Returns: (masked_sar_db, sea_binary_mask)
        """
        try:
            from app.services.land_sea_mask_service import LandSeaMaskService
            mask_svc = LandSeaMaskService.get_instance()
            mask_res = mask_svc.generate_ocean_mask(
                raster_shape=sar_db.shape[:2],
                bounds=bounds,
                transform=transform,
                sar_raster=sar_db
            )
            sea_mask = mask_res.ocean_mask
        except Exception as e:
            # Fallback radiometric thresholding
            sea_mask = (sar_db < land_threshold_db).astype(np.uint8)
            if morph_cleanup:
                sea_mask = median_filter(sea_mask, size=3)

        # Set land/nodata pixels to neutral zero / background
        masked_sar = np.where(sea_mask == 1, sar_db, -100.0)
        return masked_sar, sea_mask

    @staticmethod
    def calculate_damping_ratio(sar_db: np.ndarray, sea_mask: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates capillary wave damping depression relative to ambient sea clutter.
        Returns: (ocean_mean_db, min_slick_db, damping_delta_db)
        """
        valid_pixels = sar_db[sea_mask == 1]
        if len(valid_pixels) == 0:
            return -14.0, -14.0, 0.0

        ocean_mean_db = float(np.percentile(valid_pixels, 65))
        min_slick_db = float(np.percentile(valid_pixels, 5))
        damping_delta = min_slick_db - ocean_mean_db
        return ocean_mean_db, min_slick_db, damping_delta
