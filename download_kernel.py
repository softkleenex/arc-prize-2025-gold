#!/usr/bin/env python3
"""Download Kaggle kernel using the Kaggle API"""

from kaggle.api.kaggle_api_extended import KaggleApi
import os

# Initialize API
api = KaggleApi()
api.authenticate()

# Download the kernel
kernel_slug = "kerta27/arc-compressarc-easiest-first-strategy"
output_path = "easiest_first_kernel"

print(f"Downloading kernel: {kernel_slug}")
os.makedirs(output_path, exist_ok=True)

# Pull the kernel
api.kernels_pull(kernel_slug, path=output_path)

print(f"✓ Kernel downloaded to {output_path}/")
