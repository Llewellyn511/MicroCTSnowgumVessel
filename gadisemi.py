"""
Batch process all sample folders: segment slices in memory, stitch vessels, and fit cylinders.
Saves one combined vessels CSV and a fits.csv and report_card.json per sample.

Usage:
    - Set PARENT_INPUT_DIR and OUTPUT_PARENT_DIR as needed.
    - Run: python batch_segment_and_fit.py
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass, binary_erosion, find_objects
from scipy.ndimage import distance_transform_edt as edt
from scipy.signal import find_peaks
import imageio.v2 as iio
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, convolve

# -----------------------
# Paths
# -----------------------
PARENT_INPUT_DIR = "/scratch/w09/la5576/Tube2/InputSlices"
OUTPUT_PARENT_DIR = "/scratch/w09/la5576/Tube2/OutputCSVs"
os.makedirs(OUTPUT_PARENT_DIR, exist_ok=True)

# -----------------------
# Helpers & constants
# -----------------------
structure_2d = np.ones((3, 3), dtype=int)
structure_2d[0, 0] = structure_2d[0, 2] = structure_2d[2, 0] = structure_2d[2, 2] = 0
EPS = 1e-12
voxel_size = 0.0055655064061284065

def extract_z_from_filename(fname):
    """
    Extract integer Z value from filenames like:
        S1_z00000.tif
        S1_z00123.tiff
        sample_S1_z04567.png
    """
    m = re.search(r"_z(\d+)", fname)
    if not m:
        raise ValueError(f"Could not extract z from filename: {fname}")
    return int(m.group(1))


def compute_hist_peaks_valleys(hist, bins, peaks_idx):
    """Return valley indices/values + depths between adjacent peaks."""
    res = []
    if len(peaks_idx) < 2:
        return res
    for i in range(len(peaks_idx) - 1):
        L = peaks_idx[i]
        R = peaks_idx[i + 1]
        if R <= L + 1:
            continue
        slice_hist = hist[L:R]
        vrel = int(np.argmin(slice_hist))
        v = L + vrel
        hv = int(hist[v])
        hl = int(hist[L])
        hr = int(hist[R])
        res.append({
            "left_peak_idx": int(L),
            "right_peak_idx": int(R),
            "valley_idx": int(v),
            "valley_value": int(hv),
            "depth_left": int(hl - hv),
            "depth_right": int(hr - hv)
        })
    return res


def fit_points_direct(pts, vessel_id):
    """Fit cylinder axis and radius from pts (N,3) - returns dict including z-slices present."""
    pts = np.asarray(pts, dtype=float)

    if pts.size == 0:
        return {
            "vessel_id": int(vessel_id),
            "size": 0,
            "z_present": [],
            "z_span": 0
        }

    # --------------------------------------------
    # NEW: extract z-slices
    # --------------------------------------------
    z_all = pts[:, 2].astype(int)   # convert 1.0 → 1, assuming pts are in (x, y, z)
    z_unique = np.unique(z_all)
    z_slice_list = z_unique.tolist()
    z_span = len(z_unique)
    # --------------------------------------------

    p0 = pts.mean(axis=0)
    A = pts - p0
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    d = Vt[0]

    proj_lengths = (pts - p0) @ d
    proj_pts = p0 + np.outer(proj_lengths, d)
    r_all = np.linalg.norm(pts - proj_pts, axis=1)

    threshold = np.percentile(r_all, 95) if r_all.size > 0 else 0.0
    boundary_mask = r_all >= threshold

    if np.any(boundary_mask):
        boundary_pts = pts[boundary_mask]
        proj_boundary = proj_pts[boundary_mask]
        r_boundary = np.linalg.norm(boundary_pts - proj_boundary, axis=1)
        R = float(np.median(r_boundary))
        radial_std = float(np.std(r_boundary))
    else:
        R = 0.0
        radial_std = 0.0

    return {
        "vessel_id": int(vessel_id),
        "size": int(pts.shape[0]),
        "p0_x": float(p0[0]), "p0_y": float(p0[1]), "p0_z": float(p0[2]),
        "d_x": float(d[0]), "d_y": float(d[1]), "d_z": float(d[2]),
        "tilt_x": float(Vt[0][0]), "tilt_y": float(Vt[0][1]), "tilt_z": float(Vt[0][2]),
        "radius": R,
        "radial_std": radial_std,
        "diameter": 2.0 * R,
        "area": float(np.pi * R * R),
        "z_present": z_slice_list,
        "z_span": z_span
    }



# -----------------------
# Filtering & matching (existing functions)
# -----------------------
def filter_labels_in_slice_vectorised(label_img, min_pixels=5, max_pixels=1000, boundary_thresh=2, verbose=False):
    labels = np.unique(label_img)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return label_img.copy()
    out = np.zeros_like(label_img)
    slices = find_objects(label_img)
    for lab, sl in zip(labels, slices):
        mask = label_img[sl] == lab
        n_pix = mask.sum()
        if not (min_pixels <= n_pix <= max_pixels):
            continue
        cy, cx = center_of_mass(mask)
        boundary = mask ^ binary_erosion(mask)
        bys, bxs = np.where(boundary)
        if len(bxs) == 0:
            continue
        d = np.sqrt((bxs - cx) ** 2 + (bys - cy) ** 2)
        if np.min(d) <= boundary_thresh:
            continue
        
        out[sl][mask] = lab

    return out


def match_labels_best_overlap_vectorised(lab_z, lab_z1):
    # Flatten arrays
    a = lab_z.ravel()
    b = lab_z1.ravel()

    # Exclude background
    mask = (a != 0) & (b != 0)
    a = a[mask]
    b = b[mask]

    if a.size == 0:
        return {}

    # Remap labels to consecutive IDs
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)

    # Precompute size
    nA = len(ua)
    nB = len(ub)

    # Vectorised overlap via bincount
    overlap = np.bincount(
        ia * nB + ib,
        minlength=nA * nB
    ).reshape(nA, nB)

    # Best match for every label in A
    j = overlap.argmax(axis=1)
    v = overlap.max(axis=1)

    # Build mapping (ignore v=0)
    return {int(ua[i]): int(ub[j[i]]) for i in range(nA) if v[i] > 0}


def match_labels_best_overlap_vectorised_injectivity_tiles(
        lab_z, lab_z1,
        group_z, group_z1     # <-- NEW: group-per-pixel arrays
    ):
    """
    Vectorised maximum-overlap injective label matching between two slices,
    augmented with a grouping constraint: labels are only allowed to match
    if ALL their pixels belong to the same grouping.

    lab_z, lab_z1: 2D integer label fields (0 = background)
    group_z, group_z1: 2D integer group-id fields, same shape
    """

    # Flatten arrays
    a = lab_z.ravel()
    b = lab_z1.ravel()
    gz = group_z.ravel()
    gz1 = group_z1.ravel()

    # Exclude background
    mask = (a != 0) & (b != 0)
    a = a[mask]
    b = b[mask]
    gz = gz[mask]
    gz1 = gz1[mask]

    # If nothing remains
    if a.size == 0 or b.size == 0:
        return {}

    # Remap labels to consecutive IDs
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    nA = len(ua)
    nB = len(ub)

    # Compute overlap via bincount
    overlap = np.bincount(
        ia * nB + ib,
        minlength=nA * nB
    ).reshape(nA, nB)

    if overlap.size == 0:
        return {}

    # -------------------------------------------
    # 1) Compute group of each label (vectorised)
    # -------------------------------------------
    # For each label L in ua, build mask: a == L
    # But we do this without Python loops:
    # group of label A = mode of its group pixels

    # Map a → ia positions
    # group per label using bincount trick

    # For groups in slice z
    max_group_id = max(gz.max(), gz1.max()) + 1

    # Count group occurrences for slice z
    grpA_counts = np.zeros((nA, max_group_id), dtype=np.int64)
    np.add.at(grpA_counts, (ia, gz), 1)
    groupA = grpA_counts.argmax(axis=1)   # size nA

    # Same for slice z+1
    grpB_counts = np.zeros((nB, max_group_id), dtype=np.int64)
    np.add.at(grpB_counts, (ib, gz1), 1)
    groupB = grpB_counts.argmax(axis=1)   # size nB

    # -------------------------------------------
    # 2) Apply the grouping constraint (vectorised)
    # -------------------------------------------

    # Build a matrix of allowed matches: True where groups match
    # groupA[:, None] gives shape (nA, 1)
    # groupB[None, :] gives shape (1, nB)
    allowed = (groupA[:, None] == groupB[None, :])

    # Zero out overlaps where groups differ
    overlap *= allowed

    # If nothing is allowed, return empty map
    if overlap.max() == 0:
        return {}

    # -------------------------------------------
    # 3) Hungarian matching (same as before)
    # -------------------------------------------
    cost_matrix = -overlap
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping, only keeping positive overlap matches
    mapping = {
        int(ua[i]): int(ub[j])
        for i, j in zip(row_ind, col_ind)
        if overlap[i, j] > 0
    }

    return mapping

def match_labels_best_overlap_vectorised_injectivity(lab_z, lab_z1):


    # Flatten arrays
    a = lab_z.ravel()
    b = lab_z1.ravel()

    # Exclude background
    mask = (a != 0) & (b != 0)
    a = a[mask]
    b = b[mask]

    # Remap labels to consecutive IDs
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)

    nA = len(ua)
    nB = len(ub)

    # Vectorized overlap via bincount
    overlap = np.bincount(ia * nB + ib, minlength=nA * nB).reshape(nA, nB)

    # Only keep non-zero overlaps
    if overlap.size == 0:
        return {}

    # Convert to cost matrix for linear_sum_assignment
    # Hungarian algorithm finds the minimum cost, so negate overlaps
    cost_matrix = -overlap
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping, only keep positive overlap matches
    mapping = {int(ua[i]): int(ub[j]) for i, j in zip(row_ind, col_ind) if overlap[i, j] > 0}

    return mapping

def match_labels_6connect_vectorised(lab_z, lab_z1):
    H, W = lab_z.shape
    vol = np.zeros((H, W, 3), dtype=int)
    vol[:, :, 0] = lab_z
    vol[:, :, 1] = lab_z1
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    lab3d, _ = label(vol, structure=structure)
    coords = np.column_stack(np.nonzero(lab3d))
    labels3d = lab3d[coords[:, 0], coords[:, 1], coords[:, 2]]
    zs = coords[:, 2]
    mask_both = np.zeros(lab3d.max() + 1, dtype=bool)
    for comp_id in np.unique(labels3d):
        if comp_id == 0:
            continue
        comp_zs = zs[labels3d == comp_id]
        if 0 in comp_zs and 1 in comp_zs:
            mask_both[comp_id] = True
    valid_comps = np.nonzero(mask_both)[0]
    if valid_comps.size == 0:
        return {}
    mapping = {}
    for comp_id in valid_comps:
        coords_comp = coords[labels3d == comp_id]
        zs_comp = coords_comp[:, 2]
        z0_coords = coords_comp[zs_comp == 0][:, :2]
        z1_coords = coords_comp[zs_comp == 1][:, :2]
        lbl0 = np.unique(lab_z[z0_coords[:, 0], z0_coords[:, 1]])
        lbl1 = np.unique(lab_z1[z1_coords[:, 0], z1_coords[:, 1]])
        lbl0 = lbl0[lbl0 != 0]
        lbl1 = lbl1[lbl1 != 0]
        if len(lbl0) == 1 and len(lbl1) == 1:
            mapping[int(lbl0[0])] = int(lbl1[0])
    return mapping

def find_all_roots(mappings):
    """
    Returns:
        roots: list of (start_z, label)
    """
    roots = []

    prev_values = set()
    for z, m in enumerate(mappings):
        keys = set(m.keys())
        z_roots = keys - prev_values
        for r in z_roots:
            roots.append((z, r))
        prev_values = set(m.values())

    return roots


def filter_associations_min_presence(mappings, min_slices=20):
    n_maps = len(mappings)
    if n_maps == 0:
        return [], []

    roots = find_all_roots(mappings)
    num_chains = len(roots)

    # chain_array[i, z] = label at slice z (or nan)
    chain_array = np.full((num_chains, n_maps + 1), np.nan)

    # initialize roots at their start z
    for i, (z0, root) in enumerate(roots):
        chain_array[i, z0] = root

    # forward propagation (same logic as yours)
    for z in range(n_maps):
        mapping = mappings[z]
        curr = chain_array[:, z]

        # vectorized lookup via mask
        valid = ~np.isnan(curr)
        curr_int = curr[valid].astype(int)

        next_vals = np.array(
            [mapping.get(c, np.nan) for c in curr_int],
            dtype=float
        )

        chain_array[valid, z + 1] = next_vals

    # count presence (no restarting)
    presence_count = np.sum(~np.isnan(chain_array), axis=1)
    keep_mask = presence_count >= min_slices

    filtered_chains = chain_array[keep_mask].tolist()

    # rebuild filtered mappings
    filtered_mappings = []
    for z in range(n_maps):
        m = {}
        for chain in filtered_chains:
            if not np.isnan(chain[z]) and not np.isnan(chain[z + 1]):
                m[int(chain[z])] = int(chain[z + 1])
        filtered_mappings.append(m)

    return filtered_mappings, filtered_chains



# -----------------------
# Segmentation: returns segmented slices and stats
# -----------------------

def flip_phase0_if_majority1(img, required=7):
    # force binary 0/1   (255→1, 0→0)
    img = (img > 0).astype(np.uint8)

    kernel = np.ones((3,3), dtype=int)
    kernel[1,1] = 0  # REMOVE center pixel

    # count white neighbors (1's)
    white_neighbors = convolve((img == 1).astype(int),
                               kernel,
                               mode='nearest')

    # condition: pixel is 0 (black) AND has >= required white neighbors
    flip_mask = (img == 0) & (white_neighbors >= required)

    out = img.copy()
    out[flip_mask] = 1  # flip black → white
    return out

def segment_slices_in_memory(slice_files, folder):
    """
    Segments slices and returns:
      - segmented_slices: list of binary arrays (uint8)
      - slice_stats: list of dicts per slice (sapwood/opened areas etc.)
      - seg_report: summary of histogram thresholding and glass counts (from first slice)
    """
    segmented_slices = []
    slice_stats = []

    # Compute thresholds from first slice
    first_img = iio.imread(os.path.join(folder, slice_files[0]))
    pixels = first_img[first_img > 0].ravel()
    hist, bins = np.histogram(pixels, bins=4096, range=(1, 20000))
    peaks, _ = find_peaks(hist, prominence=2000, distance=200)
    thresholds = []
    for i in range(len(peaks) - 1):
        L, R = peaks[i], peaks[i + 1]
        valley = np.argmin(hist[L:R]) + L
        thresholds.append(bins[valley])
    t_large = max(thresholds) if thresholds else 14000 #fixing attempt with artefact
    t_small = min(thresholds) if thresholds else 11500 #fixing attempt with artefact

    # reporting summary from first slice
    seg_report = {
        "threshold_large": float(t_large),
        "threshold_small": float(t_small),
        "hist_peaks_indices": peaks.tolist(),
        "hist_peaks_values": hist[peaks].tolist(),
        "peak_distances": np.diff(peaks).tolist() if len(peaks) > 1 else [],
        "valley_bins": [float(bins[L + np.argmin(hist[L:R])]) for (L, R) in zip(peaks[:-1], peaks[1:])],
        "valley_freqs": [int(np.min(hist[L:R])) for (L, R) in zip(peaks[:-1], peaks[1:])]
    }

    glass_pixels = int((first_img >= 14000).sum()) #glass_pixels = int((first_img >= t_large).sum())
    total_pixels = int(first_img.size)
    seg_report["glass_pixel_fraction"] = float(glass_pixels / total_pixels) if total_pixels > 0 else 0.0
    seg_report["glass_pixel_count"] = glass_pixels
    seg_report["total_pixels"] = total_pixels

    r = 100  # structuring radius (unchanged behaviour)

    for fname in slice_files:
        img = iio.imread(os.path.join(folder, fname))
        GvE = (img >= 14000) #GvE = (img >= t_large) # normally but we are running the bandaid of global thresholds
        s1 = np.ones((10, 10), dtype=bool)
        CGvE = binary_closing(GvE, structure=s1)
        DCGvE = binary_dilation(CGvE, iterations=50)
        mask1 = img * (1 - DCGvE)

        AvE = ~(mask1 >= 11500) #AvE = ~(mask1 >= t_small) #global thresholds

        # Opening
        dist1 = edt(AvE)
        eroded = dist1 > r
        dist2 = edt(~eroded)
        opened = dist2 <= r
        opened = opened.astype(bool)

        # Closing
        dist3 = edt(~opened)
        dilated = dist3 <= r
        dist4 = edt(dilated)
        closed = dist4 > r
        closed = closed.astype(bool)
        COAvE = closed

        mask2 = img * (~COAvE)
        seg = ~(mask2 >= 11500) #seg = ~(mask2 >= t_small)# global threshold
        opened_white = binary_opening(seg > 0, structure=np.ones((3,3)))

        closed_white = binary_closing(opened_white, structure=np.ones((3,3)))

        seg=flip_phase0_if_majority1(closed_white, required=2)

        segmented_slices.append(seg.astype(np.uint8))

        # Compute areas (black pixels) for opened and closed-opened masks
        area_opened_AvE = int(np.count_nonzero(~opened))
        area_closed_opened_AvE = int(np.count_nonzero(~COAvE))
        slice_stats.append({
            "slice_file": fname,
            "area_opened_AvE": area_opened_AvE,
            "area_closed_opened_AvE": area_closed_opened_AvE
        })

    return segmented_slices, slice_stats, seg_report



def collate_sample_jsons(output_parent_dir):
    """
    Scan all sample folders, read their report_card.json, and collate a summary CSV.
    All stats are derived only from the JSON.
    """
    sample_folders = sorted([f for f in os.listdir(output_parent_dir)
                             if os.path.isdir(os.path.join(output_parent_dir, f))])
    summary_list = []

    for sample in sample_folders:
        json_path = os.path.join(output_parent_dir, sample, "report_card.json")
        if not os.path.exists(json_path):
            print(f"Skipping {sample}, no report_card.json found.")
            continue

        with open(json_path, "r") as f:
            j = json.load(f)

        # ---------------------------
        # Segmentation
        # ---------------------------
        seg = j.get("segmentation", {})
        hist_peak_distances = seg.get("peak_distances", [])
        hist_valley_freqs = seg.get("valley_freqs", [])

        # ---------------------------
        # Filtering
        # ---------------------------
        filtering = j.get("filtering", {}).get("per_slice", [])
        if filtering:
            arr = np.array([
                [s["n_labels_initial"], s["n_labels_after"],
                 s["n_removed"], s["n_removed_small"], s["n_removed_large"]]
                for s in filtering
            ], dtype=float)
            n_initial, n_after, n_removed, n_removed_small, n_removed_large = arr.T
            n_removed_min_radius = n_removed - n_removed_small - n_removed_large

            filtering_stats = {
                "n_initial_mean": n_initial.mean(),
                "n_initial_median": np.median(n_initial),
                "n_initial_min": n_initial.min(),
                "n_initial_max": n_initial.max(),
                "n_initial_std": n_initial.std(),

                "n_after_mean": n_after.mean(),
                "n_after_median": np.median(n_after),
                "n_after_min": n_after.min(),
                "n_after_max": n_after.max(),
                "n_after_std": n_after.std(),

                "n_removed_mean": n_removed.mean(),
                "n_removed_median": np.median(n_removed),
                "n_removed_min": n_removed.min(),
                "n_removed_max": n_removed.max(),
                "n_removed_std": n_removed.std(),

                "n_removed_small_mean": n_removed_small.mean(),
                "n_removed_small_median": np.median(n_removed_small),
                "n_removed_small_min": n_removed_small.min(),
                "n_removed_small_max": n_removed_small.max(),
                "n_removed_small_std": n_removed_small.std(),

                "n_removed_large_mean": n_removed_large.mean(),
                "n_removed_large_median": np.median(n_removed_large),
                "n_removed_large_min": n_removed_large.min(),
                "n_removed_large_max": n_removed_large.max(),
                "n_removed_large_std": n_removed_large.std(),

                "n_removed_min_radius_mean": n_removed_min_radius.mean(),
                "n_removed_min_radius_median": np.median(n_removed_min_radius),
                "n_removed_min_radius_min": n_removed_min_radius.min(),
                "n_removed_min_radius_max": n_removed_min_radius.max(),
                "n_removed_min_radius_std": n_removed_min_radius.std()
            }
        else:
            filtering_stats = {k: None for k in [
                "n_initial_mean","n_initial_median","n_initial_min","n_initial_max","n_initial_std",
                "n_after_mean","n_after_median","n_after_min","n_after_max","n_after_std",
                "n_removed_mean","n_removed_median","n_removed_min","n_removed_max","n_removed_std",
                "n_removed_small_mean","n_removed_small_median","n_removed_small_min","n_removed_small_max","n_removed_small_std",
                "n_removed_large_mean","n_removed_large_median","n_removed_large_min","n_removed_large_max","n_removed_large_std",
                "n_removed_min_radius_mean","n_removed_min_radius_median","n_removed_min_radius_min","n_removed_min_radius_max","n_removed_min_radius_std"
            ]}

        # ---------------------------
        # Vessels per slice
        # ---------------------------
        vessels_per_slice = np.array(j.get("vessel_counts_per_slice", []), dtype=float)
        if vessels_per_slice.size > 0:
            vessels_stats = {
                "vessels_mean": vessels_per_slice.mean(),
                "vessels_median": np.median(vessels_per_slice),
                "vessels_min": vessels_per_slice.min(),
                "vessels_max": vessels_per_slice.max(),
                "vessels_std": vessels_per_slice.std()
            }
        else:
            vessels_stats = {k: None for k in ["vessels_mean","vessels_median","vessels_min","vessels_max","vessels_std"]}

        # ---------------------------
        # Cylinder fitting radius stats
        # ---------------------------
        cylinder = j.get("cylinder_fitting", {})
        radius_summary = cylinder.get("radius_summary", {})
        tilt_ratios = cylinder.get("tilt_ratios", [])  # should be list of per-vessel tilts if available
        n_rejected_tilt = cylinder.get("rejected_non_vertical")

        n_accepted_tilt=cylinder.get("kept")
        
        if tilt_ratios:
            tilt_median = float(np.median(tilt_ratios))
            tilt_std = float(np.std(tilt_ratios))
        else:
            tilt_median = None
            tilt_std = None

        tilt_mean = cylinder.get("tilt_ratio_mean")
        tilt_min = cylinder.get("tilt_ratio_min")
        tilt_max = cylinder.get("tilt_ratio_max")

        # ---------------------------
        # n_slices_present stats (NEW)
        # ---------------------------

        # Extract from association.vessels list
        vessels_assoc = j.get("association", {}).get("vessels", [])

        n_slices_present_vals = [
            v.get("n_slices_present")
            for v in vessels_assoc
            if v.get("n_slices_present") is not None
        ]

        if len(n_slices_present_vals) > 0:
            n_slices_present_vals = np.array(n_slices_present_vals, dtype=float)
            slices_present_stats = {
                "n_slices_present_mean": float(n_slices_present_vals.mean()),
                "n_slices_present_median": float(np.median(n_slices_present_vals)),
                "n_slices_present_min": float(n_slices_present_vals.min()),
                "n_slices_present_max": float(n_slices_present_vals.max()),
                "n_slices_present_std": float(n_slices_present_vals.std())
            }
        else:
            slices_present_stats = {
                "n_slices_present_mean": None,
                "n_slices_present_median": None,
                "n_slices_present_min": None,
                "n_slices_present_max": None,
                "n_slices_present_std": None
            }




        #tilt_analysis

        vessels = j.get("cylinder_fitting", {}).get("per_vessel_results", [])
        if vessels:
            vectors = np.array([[v["d_x"], v["d_y"], v["d_z"]] for v in vessels], dtype=float)
            
            # Normalize each vector
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            unit_vectors = vectors / norms

            # Mean unit vector
            mean_vector = unit_vectors.mean(axis=0)
            mean_vector /= np.linalg.norm(mean_vector) + 1e-12

            # Alignment: dot product between each vector and mean vector
            alignment_values = np.dot(unit_vectors, mean_vector)
            alignment_score = float(np.mean(alignment_values))  # single scalar in [0,1]
        else:
            alignment_score = None



        # ---------------------------
        # Assemble summary
        # ---------------------------
        sample_summary = {
            "sample": sample,
            "threshold_wood": seg.get("threshold_small"),
            "threshold_glass": seg.get("threshold_large"),
            "glass_pixel_count": seg.get("glass_pixel_count"),
            "glass_pixel_fraction": seg.get("glass_pixel_fraction"),
            "hist_peak_distances": hist_peak_distances,
            "hist_valley_freqs": hist_valley_freqs,
            "n_slices": j.get("n_slices"),
            "chains_remaining": j.get("presence_filter", {}).get("chains_remaining"),
            "min_z_required": j.get("presence_filter", {}).get("min_z_required")
        }

        sample_summary.update(filtering_stats)
        sample_summary.update(vessels_stats)
        sample_summary.update(slices_present_stats)
        sample_summary.update({
            "radius_mean": radius_summary.get("mean"),
            "radius_median": radius_summary.get("median"),
            "radius_min": radius_summary.get("min"),
            "radius_max": radius_summary.get("max"),
            "radius_std": radius_summary.get("std"),
            "tilt_mean": tilt_mean,
            "tilt_median": tilt_median,
            "tilt_std": tilt_std,
            "tilt_min": tilt_min,
            "tilt_max": tilt_max,
            "n_rejected_tilt": n_rejected_tilt,
            "n_accepted_tilt": n_accepted_tilt,
            "tilt_alignment_score": alignment_score
            
            
        })

        summary_list.append(sample_summary)

    df_summary = pd.DataFrame(summary_list)
    return df_summary





# -----------------------
# Pipeline (in-memory vessel assembly + fitting)
# -----------------------
def run_pipeline_from_slices(
    segmented_slices,
    slice_stats,
    seg_report,
    sample_output_dir,
    z_values,
    verbose=True
):
    """
    Memory-efficient pipeline that writes vessel CSVs immediately, 
    keeps reporting, and fits cylinders per vessel.
    """

    assert len(segmented_slices) == len(z_values), "Z-values and slices length mismatch"


    n_slices=len(segmented_slices)
    report = {
        "segmentation": seg_report,
        "filtering": {"per_slice": []},
        "association": {"vessels": []},
        "presence_filter": {},
        "writing": {},
        "cylinder_fitting": {}
    }

    report["n_slices"] = n_slices

    os.makedirs(sample_output_dir, exist_ok=True)
    vessel_dir = os.path.join(sample_output_dir, "vessels")
    fit_dir = os.path.join(sample_output_dir, "fits")
    os.makedirs(vessel_dir, exist_ok=True)
    os.makedirs(fit_dir, exist_ok=True)

    # Save sapwood stats
    sapwood_csv = os.path.join(sample_output_dir, "sapwood.csv")
    pd.DataFrame(slice_stats).to_csv(sapwood_csv, index=False)
    report["sapwood_csv"] = sapwood_csv
    if verbose:
        print(f"Saved sapwood area per slice to {sapwood_csv}")

    n_slices = len(segmented_slices)
    labeled_slices = []
    filtered_slices = []
    vessels_per_slice = []

    # -------------------------
    # Per-slice labeling and filtering
    # -------------------------
    for i, arr in enumerate(segmented_slices):
        bw = (arr > 0)
        lab, _ = label(bw.astype(np.uint8), structure=structure_2d)
        labeled_slices.append(lab)

        # Filter labels
        filtered = filter_labels_in_slice_vectorised(lab, min_pixels=5, max_pixels=1000, boundary_thresh=1, verbose=False)
        filtered_slices.append(filtered)
        kept_labels = np.unique(filtered[filtered>0])
        vessels_per_slice.append(len(kept_labels))

        # Slice stats for report
        labels_raw = np.unique(lab)
        labels_raw = labels_raw[labels_raw != 0]
        n_labels_initial = len(labels_raw)
        n_labels_after = len(kept_labels)
        removed_set = set(labels_raw.tolist()) - set(kept_labels.tolist())
        sizes = np.bincount(lab.ravel(), minlength=int(lab.max())+1)[labels_raw] if n_labels_initial > 0 else np.array([], dtype=int)
        n_removed_small = int(np.sum(sizes[np.isin(labels_raw, list(removed_set))] < 5)) if sizes.size > 0 else 0
        n_removed_large = int(np.sum(sizes[np.isin(labels_raw, list(removed_set))] > 1000)) if sizes.size > 0 else 0

        report["filtering"]["per_slice"].append({
            "slice_index": i,
            "n_labels_initial": n_labels_initial,
            "mean_size_before": float(sizes.mean()) if sizes.size > 0 else 0.0,
            "median_size_before": float(np.median(sizes)) if sizes.size > 0 else 0.0,
            "n_labels_after": n_labels_after,
            "n_removed": len(removed_set),
            "n_removed_small": n_removed_small,
            "n_removed_large": n_removed_large
        })

    if verbose:
        print(f"Number of vessels per slice after filtering: {vessels_per_slice}")

    # -------------------------
    # Slice-to-slice mappings
    # -------------------------
    mappings = [match_labels_best_overlap_vectorised_injectivity(filtered_slices[z], filtered_slices[z+1]) 
                for z in range(n_slices-1)]

    filtered_mappings, chains = filter_associations_min_presence(mappings, min_slices=min(20,n_slices))
    report["presence_filter"]["chains_remaining"] = len(chains)
    report["presence_filter"]["min_z_required"] = min(5,n_slices)
    if verbose:
        print(f"Chains spanning at least min_slices: {len(chains)}")

    # -------------------------
    # Vessel CSV writing and reporting
    # -------------------------
    vessel_assoc_reports = []
    vessel_csv_paths = []

    for vidx, chain in enumerate(chains, start=1):
        coords_list = []
        slice_centroids = []
        slice_radii = []

        for z_idx, lbl in enumerate(chain):
            if np.isnan(lbl):
                continue

            lab_id = int(lbl)
            mask = (filtered_slices[z_idx] == lab_id)
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue

            true_z = z_values[z_idx]
            zs = np.full_like(xs, true_z, dtype=int)

            coords_list.append(np.column_stack([xs, ys, zs]))


            cx, cy = xs.mean(), ys.mean()
            slice_centroids.append((cx, cy))
            slice_radii.append(np.sqrt(xs.size/np.pi))
        
        label_chain = {
            int(z): int(lbl)
            for z, lbl in enumerate(chain)
            if not np.isnan(lbl)
        }


        if len(coords_list) == 0:
            continue  # skip empty vessel

        coords = np.vstack(coords_list)
        vessel_csv_path = os.path.join(vessel_dir, f"vessel_{vidx}.csv")
        pd.DataFrame(coords, columns=["x","y","z"]).to_csv(vessel_csv_path, index=False)
        vessel_csv_paths.append((vidx, vessel_csv_path))

        vessel_assoc_reports.append({
            "vessel_id": vidx,
            "label_chain": label_chain,
            "radius_min": float(np.min(slice_radii)) if slice_radii else None,
            "radius_max": float(np.max(slice_radii)) if slice_radii else None,
            "radius_mean": float(np.mean(slice_radii)) if slice_radii else None,
            "centroid_variation": float(np.mean(np.std(np.array(slice_centroids), axis=0))) if len(slice_centroids) > 1 else 0.0,
            "n_slices_present": len(slice_radii)
        })

        if verbose:
            print(f"Wrote vessel {vidx}: {coords.shape[0]} voxels -> {vessel_csv_path}")

    report["association"]["vessels"] = vessel_assoc_reports
    report["writing"]["vessels_written"] = len(vessel_assoc_reports)

    # -------------------------
    # Cylinder fitting (parallel)
    # -------------------------
    results = []
    rejected = 0

    def run_fit_item_csv(item):
        vid, csv_path = item
        pts = pd.read_csv(csv_path)[["x","y","z"]].to_numpy(dtype=float)
        return fit_points_direct(pts, int(vid))

    with ThreadPoolExecutor(max_workers=4) as executor:
        for res in executor.map(run_fit_item_csv, vessel_csv_paths):
            dx, dy, dz = abs(res.get("d_x",0.0)), abs(res.get("d_y",0.0)), abs(res.get("d_z",0.0))
            if dz > 2*dx and dz > 2*dy:
                results.append(res)
            else:
                rejected += 1

    # Save per-vessel results for summary
    report["cylinder_fitting"]["per_vessel_results"] = results


    report["cylinder_fitting"]["kept"] = len(results)
    report["cylinder_fitting"]["rejected_non_vertical"] = rejected
    report["cylinder_fitting"]["total_fitted"] = len(results)+rejected

    if results:
        tilt_ratios = [abs(r["d_z"])/(abs(r["d_x"])+abs(r["d_y"])+1e-6) for r in results]
        report["cylinder_fitting"].update({
            "tilt_ratio_mean": float(np.mean(tilt_ratios)),
            "tilt_ratio_min": float(np.min(tilt_ratios)),
            "tilt_ratio_max": float(np.max(tilt_ratios))
        })
    else:
        report["cylinder_fitting"].update({
            "tilt_ratio_mean": None, "tilt_ratio_min": None, "tilt_ratio_max": None
        })
    
    

    # -------------------------
    # Aggregate radius + tilt statistics (sample-level)
    # -------------------------
    if results:
        # ---- Radius summary ----
        radii = np.array([r["radius"] for r in results if r.get("radius") is not None])

        report["cylinder_fitting"]["radius_summary"] = {
            "mean": float(radii.mean()) if radii.size else None,
            "median": float(np.median(radii)) if radii.size else None,
            "min": float(radii.min()) if radii.size else None,
            "max": float(radii.max()) if radii.size else None,
            "std": float(radii.std()) if radii.size else None
        }

        # ---- Tilt ratio summary ----
        tilt_ratios = np.array([
            abs(r["d_z"]) / (abs(r["d_x"]) + abs(r["d_y"]) + 1e-6)
            for r in results
        ])

        report["cylinder_fitting"].update({
            "tilt_ratios": tilt_ratios.tolist(),
            "tilt_ratio_mean": float(tilt_ratios.mean()),
            "tilt_ratio_median": float(np.median(tilt_ratios)),
            "tilt_ratio_std": float(tilt_ratios.std()),
            "tilt_ratio_min": float(tilt_ratios.min()),
            "tilt_ratio_max": float(tilt_ratios.max())
        })

        # ---- Mean axis + angular dispersion ----
        dirs = np.array([[r["d_x"], r["d_y"], r["d_z"]] for r in results], dtype=float)

        # Enforce same hemisphere (z positive)
        dirs[dirs[:, 2] < 0] *= -1

        C = np.cov(dirs.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        mean_axis = eigvecs[:, np.argmax(eigvals)]
        mean_axis /= np.linalg.norm(mean_axis)

        dots = np.clip(dirs @ mean_axis, -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))

        report["cylinder_fitting"]["axis_summary"] = {
            "mean_axis": {
                "x": float(mean_axis[0]),
                "y": float(mean_axis[1]),
                "z": float(mean_axis[2])
            },
            "angle_mean_deg": float(angles.mean()),
            "angle_median_deg": float(np.median(angles)),
            "angle_std_deg": float(angles.std()),
            "angle_min_deg": float(angles.min()),
            "angle_max_deg": float(angles.max())
        }
    else:
        report["cylinder_fitting"]["radius_summary"] = {}
        report["cylinder_fitting"]["axis_summary"] = {}
        report["cylinder_fitting"].update({
            "tilt_ratios": [],
            "tilt_ratio_mean": None,
            "tilt_ratio_median": None,
            "tilt_ratio_std": None,
            "tilt_ratio_min": None,
            "tilt_ratio_max": None
        })


    # Save fits CSV
    fits_csv_path = os.path.join(fit_dir, "fits.csv")
    df_results = pd.DataFrame(results) if results else pd.DataFrame(columns=[
        "vessel_id","size","p0_x","p0_y","p0_z","d_x","d_y","d_z",
        "tilt_x","tilt_y","tilt_z","radius","radial_std","diameter","area"
    ])
    df_results.to_csv(fits_csv_path, index=False)
    report["fits_csv"] = fits_csv_path
    if verbose:
        print(f"Saved cylinder fits to {fits_csv_path}")

    # -------------------------
    # Add summary info to JSON
    # -------------------------
    report["n_slices"] = n_slices
    report["vessel_counts_per_slice"] = vessels_per_slice


    # Save JSON report
    report_path = os.path.join(sample_output_dir, "report_card.json")
    with open(report_path,"w") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(f"Saved report card to {report_path}")

    return {
        "n_slices": n_slices,
        "vessels_written": len(vessel_assoc_reports),
        "fits_saved": len(results),
        "vessel_dir": vessel_dir,
        "fits_csv": fits_csv_path,
        "report_card": report_path
    }



# -----------------------
# Parallel batch processing
# -----------------------

def process_single_sample(sample):
    """
    Process one sample folder (top-level directory name).
    This function is executed inside a worker process.
    """
    try:
        sample_dir = os.path.join(PARENT_INPUT_DIR, sample)
        slice_files = [
            f for f in os.listdir(sample_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        slice_files.sort(key=extract_z_from_filename)

        if len(slice_files) == 0:
            return f"No slices for {sample}; skipped."

        z_values = [extract_z_from_filename(f) for f in slice_files]

        # segmentation step (unchanged)
        segmented_slices, slice_stats, seg_report = segment_slices_in_memory(
            slice_files, sample_dir
        )

        # output path for this sample
        output_dir = os.path.join(OUTPUT_PARENT_DIR, sample)

        # run your full existing pipeline
        info = run_pipeline_from_slices(
            segmented_slices,
            slice_stats,
            seg_report,
            output_dir,
            z_values=z_values,
            verbose=False,   # workers stay quiet unless you prefer True
        )

        return f"{sample} processed OK. Summary: {info}"

    except Exception as e:
        return f"ERROR in {sample}: {e}"
    

from mpi4py import MPI

def mpi_batch():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ----------------------------
    # Discover samples (only rank 0 reads the filesystem)
    # ----------------------------
    if rank == 0:
        sample_folders = [
            f for f in os.listdir(PARENT_INPUT_DIR)
            if os.path.isdir(os.path.join(PARENT_INPUT_DIR, f))
        ]
        sample_folders.sort()
    else:
        sample_folders = None

    # Broadcast the list to all ranks
    sample_folders = comm.bcast(sample_folders, root=0)

    # Split work evenly: each rank does samples[rank::size]
    my_samples = sample_folders[rank::size]

    results = []

    # ----------------------------
    # Process each assigned sample
    # ----------------------------
    for sample in my_samples:
        try:
            msg = process_single_sample(sample)
            results.append(msg)
            print(f"[Rank {rank}] {msg}", flush=True)
        except Exception as e:
            err = f"ERROR in {sample}: {e}"
            print(f"[Rank {rank}] {err}", flush=True)
            results.append(err)

    # ----------------------------
    # Gather all worker messages on rank 0
    # ----------------------------
    all_results = comm.gather(results, root=0)

    # ----------------------------
    # Final summary: ONLY rank 0
    # ----------------------------
    if rank == 0:
        flat_results = [r for sub in all_results for r in sub]
        print("\n=== MPI Processing Summary ===")
        for line in flat_results:
            print(" ", line)

        print("\nGenerating overall summary across all samples...")
        df_summary = collate_sample_jsons(OUTPUT_PARENT_DIR)
        summary_csv_path = os.path.join(OUTPUT_PARENT_DIR,
                                        "summary_all_samples.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Saved overall summary CSV to {summary_csv_path}")
        print("\nAll samples complete.")



# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    mpi_batch()
