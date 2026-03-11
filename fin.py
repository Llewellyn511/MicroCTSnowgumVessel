import os
import re
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from imageio import imread

input_folder = r"C:\CTSCANS\Analysis\InputSlices(T1)"
output_folder = r"C:\CTSCANS\Analysis\OutputCSVs(T1)final"
os.makedirs(output_folder, exist_ok=True)

voxel_size = 0.0055655064061284065

def extract_number(fname: str) -> int:
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r'(\d+)', base)
    return int(m.group(1)) if m else float('inf')

def is_image(fname: str) -> bool:
    return fname.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))

def detect_vessels_and_sapwood(img):
    pixels = img.ravel()
    pixels = pixels[pixels > 0]

    hist, bins = np.histogram(pixels, bins=4096, range=(1, 20000))
    peaks, _ = find_peaks(hist, prominence=2000, distance=200)
    thresholds = []
    for i in range(len(peaks) - 1):
        left, right = peaks[i], peaks[i+1]
        valley_index = np.argmin(hist[left:right]) + left
        thresholds.append(bins[valley_index])

    t_large = max(thresholds) if len(thresholds) > 0 else 20000
    t_small = min(thresholds) if len(thresholds) > 0 else 10000

    GvE = (img >= t_large)
    s1 = np.ones((10, 10), dtype=bool)
    CGvE = ndi.binary_closing(GvE, structure=s1)
    DCGvE = ndi.binary_dilation(CGvE, iterations=50)
    mask1 = img * (1 - DCGvE)

    AvE = 1 - (mask1 >= t_small)
    s2 = np.ones((140, 140), dtype=bool)
    OAvE = ndi.binary_opening(AvE, structure=s2)
    COAvE = ndi.binary_closing(OAvE, structure=s2)
    mask2 = img * (1 - COAvE)

    seg = 1 - (mask2 >= t_small)

    labelstructure = np.ones((3, 3), dtype=int)
    labeled, num_features = ndi.label(seg, structure=labelstructure)

    sizes = ndi.sum(seg, labeled, range(1, num_features + 1))
    sizes = np.array(sizes, dtype=int)

    filteredsizes = sizes[5 < sizes < 1000]

    vessel_area_mm2 = filteredsizes * (voxel_size ** 2)
    vessel_diam_um = 2 * np.sqrt(vessel_area_mm2 / np.pi) * 1000

    total_black = np.sum(OAvE)
    vessel_area_px = np.sum(filteredsizes)
    sapwood_pixels = total_black - vessel_area_px
    sapwood_mm2 = sapwood_pixels * (voxel_size ** 2)

    # --- New: total stem area in mm² ---
    stem_area_mm2 = sapwood_mm2 + np.sum(vessel_area_mm2)

    return vessel_diam_um, sapwood_mm2, stem_area_mm2

files = [f for f in os.listdir(input_folder) if is_image(f)]
files.sort(key=lambda f: (extract_number(f), f.lower()))

print("Files found (in order):", files)

for fname in files:
    fpath = os.path.join(input_folder, fname)
    img = imread(fpath)

    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[..., 0]
        else:
            img = img.mean(axis=-1).astype(img.dtype)

    vessel_diam_um, sapwood_mm2, stem_area_mm2 = detect_vessels_and_sapwood(img)

    outname = os.path.splitext(fname)[0] + ".csv"
    outpath = os.path.join(output_folder, outname)

    df = pd.DataFrame({
        "vessel_diameter_um": vessel_diam_um
    })
    df["sapwood_area_mm2"] = sapwood_mm2
    df["stem_area_mm2"] = stem_area_mm2  # <- Added column

    df.to_csv(outpath, index=False)
    print(f"Processed {fname} → {outname} "
          f"(count {len(vessel_diam_um)}, sapwood {sapwood_mm2:.2f} mm², "
          f"stem {stem_area_mm2:.2f} mm²)")

print(f"\nAll slices processed. CSVs saved to {output_folder}")
