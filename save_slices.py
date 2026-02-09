#!/usr/bin/env python3
import os
import shutil

# -----------------------
# CONFIGURATION
# -----------------------
SOURCE_DIR = "/scratch/w09/la5576/Tube2/Wood_stack_2_full_res_top/TIF_singles"
OUTPUT_ROOT = "/scratch/w09/la5576/Tube2/Retestslices"
RANGES_FILE = "ranges_top.txt"  # file in current directory
BIN_FACTOR = 8
# -----------------------

with open(RANGES_FILE) as f:
    for line in f:
        if not line.strip() or line.startswith("#"):
            continue

        start_str, end_str, N_str = line.strip().split()
        start = int(start_str)
        end = int(end_str)
        N = N_str.strip()

        target_folder = os.path.join(OUTPUT_ROOT, f"Sample{N}")
        os.makedirs(target_folder, exist_ok=True)

        print(f"Processing Sample{N}: slices {start}-{end}")

        slice_counter = 0  # output z numbering starts at 0

        for z in range(start, end + 1):
            src_index = z * BIN_FACTOR
            for f in range(BIN_FACTOR):
                src_file = os.path.join(SOURCE_DIR, f"TIF_{src_index + f:05d}.tif")
                dst_file = os.path.join(target_folder, f"S{N}_z{slice_counter:03d}.tif")

                if not os.path.exists(src_file):
                    print(f"WARNING: {src_file} does not exist, skipping")
                    continue

                shutil.copy2(src_file, dst_file)
                slice_counter += 1

        print(f"Sample{N} complete: {slice_counter} slices saved to {target_folder}")

