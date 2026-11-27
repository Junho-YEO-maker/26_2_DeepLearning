"""
SEM metrology: CD (line width) and LER (edge roughness) measurement.
Key metrics for evaluating if denoised SEM can be used for metrology.
"""

import numpy as np
from scipy.signal import find_peaks


def measure_edge_profile(image, nominal_x, search_width=15):
    """Measure sub-pixel edge position per row using threshold crossing."""
    h, w = image.shape
    positions = np.zeros(h)
    x_start = max(0, int(nominal_x) - search_width)
    x_end = min(w, int(nominal_x) + search_width)

    for row in range(h):
        profile = image[row, x_start:x_end].astype(np.float64)
        if len(profile) < 5:
            positions[row] = nominal_x
            continue
        pmin, pmax = profile.min(), profile.max()
        if pmax - pmin < 5:
            positions[row] = nominal_x
            continue
        mid = (pmin + pmax) / 2
        crossings = np.where(np.diff(np.sign(profile - mid)))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            if idx + 1 < len(profile):
                frac = (mid - profile[idx]) / (profile[idx + 1] - profile[idx] + 1e-10)
                positions[row] = x_start + idx + frac
            else:
                positions[row] = x_start + idx
        else:
            positions[row] = nominal_x
    return positions


def detect_edges(image, threshold=None):
    """Detect line edges using intensity gradient."""
    avg_profile = np.mean(image, axis=0)
    gradient = np.gradient(avg_profile)
    if threshold is None:
        threshold = np.std(gradient) * 1.5
    rising = find_peaks(gradient, height=threshold, distance=10)[0]
    falling = find_peaks(-gradient, height=threshold, distance=10)[0]
    edges = []
    for r in rising:
        candidates = falling[falling > r]
        if len(candidates) > 0:
            f = candidates[0]
            if f - r > 3:
                edges.append((r, f))
    return edges


def measure_cd(image, gt_edges=None):
    """Measure Critical Dimension (line width)."""
    edge_pairs = gt_edges if gt_edges else detect_edges(image)
    results = []
    for left_nom, right_nom in edge_pairs:
        left_profile = measure_edge_profile(image, left_nom)
        right_profile = measure_edge_profile(image, right_nom)
        cd_per_row = right_profile - left_profile
        results.append({
            "cd_mean": float(np.mean(cd_per_row)),
            "cd_std": float(np.std(cd_per_row)),
            "left_profile": left_profile,
            "right_profile": right_profile,
        })
    return results


def measure_ler(edge_profile):
    """Measure Line Edge Roughness: 3 * std(detrended edge positions)."""
    rows = np.arange(len(edge_profile))
    coeffs = np.polyfit(rows, edge_profile, 1)
    trend = np.polyval(coeffs, rows)
    return float(3 * np.std(edge_profile - trend))


class MetrologyEvaluator:
    """Compare CD/LER measurements between denoised and ground truth."""

    def __init__(self, gt_params):
        self.gt = gt_params

    def evaluate(self, denoised_image):
        results = {"lines": []}
        gt_edges = []
        for line_gt in self.gt["lines"]:
            left_mean = np.mean(line_gt["left_edge"])
            right_mean = np.mean(line_gt["right_edge"])
            gt_edges.append((left_mean, right_mean))

        denoised_cds = measure_cd(denoised_image, gt_edges)
        total_cd_error = 0
        total_ler_error = 0
        n = len(denoised_cds)

        for meas, gt_line in zip(denoised_cds, self.gt["lines"]):
            gt_cd = gt_line["cd_mean"]
            gt_ler_l = gt_line["ler_left_rms"] * 3
            gt_ler_r = gt_line["ler_right_rms"] * 3
            cd_err = abs(meas["cd_mean"] - gt_cd)
            ler_err_l = abs(measure_ler(meas["left_profile"]) - gt_ler_l)
            ler_err_r = abs(measure_ler(meas["right_profile"]) - gt_ler_r)
            total_cd_error += cd_err
            total_ler_error += (ler_err_l + ler_err_r) / 2
            results["lines"].append({
                "gt_cd": gt_cd, "measured_cd": meas["cd_mean"],
                "cd_error": cd_err, "ler_error": (ler_err_l + ler_err_r) / 2
            })

        results["mean_cd_error"] = total_cd_error / max(n, 1)
        results["mean_ler_error"] = total_ler_error / max(n, 1)
        results["cd_error_percent"] = results["mean_cd_error"] / self.gt["nominal_cd"] * 100
        return results
