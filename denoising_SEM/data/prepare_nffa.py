"""
NFFA-EUROPE SEM Dataset Preparation for Denoising Experiments.

Downloads real SEM images from NFFA-EUROPE (Zenodo) and prepares
paired noisy/clean datasets by adding synthetic SEM noise to real images.

Strategy:
  - Real SEM images serve as "clean" references
  - Synthetic SEM noise (Poisson + Gaussian + scanlines) is added
  - Three noise levels: low, medium, high (matching synthetic experiments)
  - Images are center-cropped and resized to 256x256
"""

import os
import sys
import glob
import random
import shutil
import hashlib
import argparse
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Download helpers
# ---------------------------------------------------------------------------

def download_nffa_subset(data_dir, max_images=200):
    """
    Download NFFA-EUROPE SEM dataset from Zenodo.
    Record ID: 7615060 (Sub-dataset: SEM, ~450MB, 577 images)

    Falls back to manual download instructions if wget fails.
    """
    import subprocess

    zip_url = "https://zenodo.org/records/7615060/files/SEM.zip?download=1"
    zip_path = os.path.join(data_dir, "SEM.zip")
    extract_dir = os.path.join(data_dir, "SEM_raw")

    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(extract_dir) and len(glob.glob(os.path.join(extract_dir, "**", "*.tif"), recursive=True)) > 0:
        print(f"[INFO] Raw SEM data already exists at {extract_dir}")
        return extract_dir

    # Try downloading
    if not os.path.exists(zip_path):
        print(f"[INFO] Downloading NFFA-EUROPE SEM dataset from Zenodo...")
        print(f"  URL: {zip_url}")
        print(f"  This may take a few minutes (~450MB)...")

        try:
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", zip_path, zip_url],
                timeout=600,  # 10 min timeout
                capture_output=False
            )
            if result.returncode != 0:
                raise RuntimeError("wget failed")
        except Exception as e:
            print(f"[WARN] Download failed: {e}")
            print(f"[INFO] Trying curl...")
            try:
                result = subprocess.run(
                    ["curl", "-L", "-o", zip_path, zip_url],
                    timeout=600,
                    capture_output=False
                )
                if result.returncode != 0:
                    raise RuntimeError("curl failed")
            except Exception as e2:
                print(f"[ERROR] Both wget and curl failed.")
                print(f"  Please download manually:")
                print(f"  1. Go to https://zenodo.org/records/7615060")
                print(f"  2. Download SEM.zip")
                print(f"  3. Place it at: {zip_path}")
                sys.exit(1)

    # Extract
    print(f"[INFO] Extracting {zip_path}...")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", extract_dir], check=True)
    except Exception:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

    print(f"[INFO] Extraction complete.")
    return extract_dir


# ---------------------------------------------------------------------------
# 2. Image preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(image_path, target_size=256):
    """
    Load a SEM image, convert to grayscale, center-crop to square,
    and resize to target_size x target_size.
    """
    try:
        img = Image.open(image_path).convert('L')  # grayscale
    except Exception as e:
        return None

    w, h = img.size

    # Skip very small images
    if min(w, h) < 128:
        return None

    # Center crop to square
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

    # Resize
    img = img.resize((target_size, target_size), Image.LANCZOS)

    return np.array(img, dtype=np.float32) / 255.0


def collect_images(raw_dir, max_images=200, target_size=256, seed=2021147017):
    """
    Collect and preprocess SEM images from the raw directory.
    Returns list of numpy arrays (float32, [0,1]).
    """
    # Find all image files
    extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))

    print(f"[INFO] Found {len(all_files)} image files in {raw_dir}")

    if len(all_files) == 0:
        print("[ERROR] No images found!")
        sys.exit(1)

    # Deterministic shuffle
    random.seed(seed)
    random.shuffle(all_files)

    images = []
    for fpath in all_files:
        if len(images) >= max_images:
            break
        img = load_and_preprocess(fpath, target_size)
        if img is not None:
            images.append(img)

    print(f"[INFO] Successfully loaded {len(images)} images")
    return images


# ---------------------------------------------------------------------------
# 3. SEM noise synthesis (matches synthetic_sem.py noise model)
# ---------------------------------------------------------------------------

def add_sem_noise(clean_img, noise_level='medium', rng=None):
    """
    Add realistic SEM noise to a clean image.
    Noise model: Poisson (shot noise) + Gaussian (detector) + scanlines

    Parameters match the synthetic experiment noise levels:
      low:    poisson_factor=200, gaussian_std=0.02, scanline_amp=0.01
      medium: poisson_factor=100, gaussian_std=0.05, scanline_amp=0.03
      high:   poisson_factor=50,  gaussian_std=0.10, scanline_amp=0.05
    """
    if rng is None:
        rng = np.random.RandomState(42)

    noise_params = {
        'low':    {'poisson_factor': 200, 'gaussian_std': 0.02, 'scanline_amp': 0.01},
        'medium': {'poisson_factor': 100, 'gaussian_std': 0.05, 'scanline_amp': 0.03},
        'high':   {'poisson_factor':  50, 'gaussian_std': 0.10, 'scanline_amp': 0.05},
    }

    params = noise_params[noise_level]
    img = clean_img.copy()

    # 1. Poisson noise (shot noise)
    pf = params['poisson_factor']
    img_scaled = np.clip(img * pf, 0, None)
    img_poisson = rng.poisson(img_scaled).astype(np.float32) / pf

    # 2. Gaussian noise (detector noise)
    gaussian = rng.normal(0, params['gaussian_std'], img.shape).astype(np.float32)
    img_noisy = img_poisson + gaussian

    # 3. Scanline noise (horizontal banding)
    h = img.shape[0]
    scanline = rng.normal(0, params['scanline_amp'], h).astype(np.float32)
    img_noisy += scanline[:, np.newaxis]

    return np.clip(img_noisy, 0, 1)


# ---------------------------------------------------------------------------
# 4. Dataset creation
# ---------------------------------------------------------------------------

def create_dataset(images, output_dir, noise_levels=('low', 'medium', 'high'),
                   train_ratio=0.8, seed=2021147017):
    """
    Create train/test splits with paired noisy/clean images for each noise level.

    Output structure:
      output_dir/
        {noise_level}/
          train/
            noisy/  img_000.png, img_001.png, ...
            clean/  img_000.png, img_001.png, ...
          test/
            noisy/  img_000.png, img_001.png, ...
            clean/  img_000.png, img_001.png, ...
    """
    n = len(images)
    n_train = int(n * train_ratio)

    # Deterministic split
    rng = np.random.RandomState(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    print(f"[INFO] Dataset split: {len(train_indices)} train, {len(test_indices)} test")

    for noise_level in noise_levels:
        print(f"[INFO] Generating {noise_level} noise pairs...")

        for split_name, split_indices in [('train', train_indices), ('test', test_indices)]:
            noisy_dir = os.path.join(output_dir, noise_level, split_name, 'noisy')
            clean_dir = os.path.join(output_dir, noise_level, split_name, 'clean')
            os.makedirs(noisy_dir, exist_ok=True)
            os.makedirs(clean_dir, exist_ok=True)

            for i, idx in enumerate(split_indices):
                clean_img = images[idx]

                # Different noise realization per image (but deterministic)
                noise_rng = np.random.RandomState(seed + idx * 7 + hash(noise_level) % 10000)
                noisy_img = add_sem_noise(clean_img, noise_level, noise_rng)

                # Save as PNG (uint8)
                clean_u8 = (clean_img * 255).astype(np.uint8)
                noisy_u8 = (noisy_img * 255).astype(np.uint8)

                fname = f"img_{i:03d}.png"
                Image.fromarray(clean_u8).save(os.path.join(clean_dir, fname))
                Image.fromarray(noisy_u8).save(os.path.join(noisy_dir, fname))

        # Count
        n_train_saved = len(os.listdir(os.path.join(output_dir, noise_level, 'train', 'noisy')))
        n_test_saved = len(os.listdir(os.path.join(output_dir, noise_level, 'test', 'noisy')))
        print(f"  → {noise_level}: {n_train_saved} train + {n_test_saved} test pairs saved")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Prepare NFFA-EUROPE SEM dataset')
    parser.add_argument('--data-dir', type=str, default='data/nffa',
                        help='Directory for raw downloads')
    parser.add_argument('--output-dir', type=str, default='data/nffa_sem',
                        help='Output directory for processed dataset')
    parser.add_argument('--max-images', type=int, default=200,
                        help='Maximum number of images to use')
    parser.add_argument('--target-size', type=int, default=256,
                        help='Target image size (square)')
    parser.add_argument('--seed', type=int, default=2021147017,
                        help='Random seed (student ID)')
    args = parser.parse_args()

    print("=" * 60)
    print("NFFA-EUROPE SEM Dataset Preparation")
    print("=" * 60)

    # Step 1: Download
    raw_dir = download_nffa_subset(args.data_dir, args.max_images)

    # Step 2: Collect and preprocess
    images = collect_images(raw_dir, args.max_images, args.target_size, args.seed)

    # Step 3: Create paired datasets
    create_dataset(images, args.output_dir, seed=args.seed)

    print()
    print("=" * 60)
    print("Dataset preparation complete!")
    print(f"  Output: {args.output_dir}")
    print(f"  Noise levels: low, medium, high")
    print(f"  Images per level: {args.max_images}")
    print("=" * 60)


if __name__ == '__main__':
    main()
