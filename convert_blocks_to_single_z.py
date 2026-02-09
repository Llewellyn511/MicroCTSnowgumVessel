#!/usr/bin/env python3
import os
import numpy as np
import tifffile
from ANUNetCDF import ANUNetCDF, ANUNetCDFType

# === INPUT / OUTPUT PATHS ===
INPUT_DIR = "/scratch/w09/la5576/Tube2/Wood_stack_2_full_res_top/tomoHiRes_nc/"
OUTPUT_DIR = "/scratch/w09/la5576/Tube2/Wood_stack_2_full_res_top/TIF_singles/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === INITIALIZE ANU READER ===
anunc = ANUNetCDF()

# === SORT BLOCKS ===
files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".nc"))

if not files:
    raise RuntimeError("No .nc files found in input directory.")

slice_counter = 1  # global counter for consecutive TIF numbering

for fname in files:
    in_path = os.path.join(INPUT_DIR, fname)
    print(f"Reading block: {fname}")
    
    # read NetCDF block
    array, origin, voxsz, voxunits, ID, histories = anunc.readFile(fn=in_path, imagetype=ANUNetCDFType.Tomo)
    
    # fill masked values
    array = np.ma.filled(array, fill_value=0)
    
    # array.shape is (X, Y, Z)
    X, Y, Z = array.shape
    
    # write each Z slice as its own TIFF
    for z in range(Z):
        slice_img = array[:, :, z]  # select slice
        out_name = os.path.join(OUTPUT_DIR, f"TIF_{slice_counter:05}.tif")
        tifffile.imwrite(out_name, slice_img)
        print(f"Wrote slice {slice_counter}: {out_name}")
        slice_counter += 1

print("All blocks converted to single-slice TIFFs!")

