"""Image quality metrics: PSNR, SSIM."""

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def compute_psnr(clean, denoised, data_range=255.0):
    return float(peak_signal_noise_ratio(clean, denoised, data_range=data_range))


def compute_ssim(clean, denoised, data_range=255.0):
    return float(structural_similarity(clean, denoised, data_range=data_range))


def compute_all(clean, denoised, data_range=255.0):
    return {"psnr": compute_psnr(clean, denoised, data_range), "ssim": compute_ssim(clean, denoised, data_range)}
