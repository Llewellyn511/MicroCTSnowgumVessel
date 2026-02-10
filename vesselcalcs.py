import pandas as pd
import numpy as np
import ast
import os

### --------------------------------------------------------
### CONSTANTS & PARAMETERS
### --------------------------------------------------------

voxel_size_mm = 0.0055655064061284065
voxel_size_m = voxel_size_mm * 1e-3     # convert mm → m
eta = 1.0e-9                             # MPa·s (viscosity of water ~20°C)

# Hydraulic parameters
s = 20                  # logistic steepness
alpha = 2.5e-5          # Sperry/Pittermann scaling constant
k = 2.1                 # exponent for P50 ~ r^{-k}

# Pressure range (MPa)
P = np.linspace(0, 3, 301)  # cavitation curve


### --------------------------------------------------------
### PATHS
### --------------------------------------------------------

root = r"C:\Users\hello\Llewellyn\Analysis\Tiltanalysis\Workingfits"
meta_xlsx = r"C:\Users\hello\Llewellyn\Analysis\Tiltanalysis\INPUTS\Snowgum microCT 2.0.xlsx"

meta = pd.read_excel(meta_xlsx, sheet_name=0, engine="openpyxl")
# Columns: tube_ID, biome_site


### --------------------------------------------------------
### PROCESS ONE SAMPLE
### --------------------------------------------------------
def process_sample(sample_name, tube_id, biome):
    folder = f"{root}\\{sample_name}"

    # ------------------------------------------------------
    # 1. Sapwood.csv → sapwood area (convert voxels → mm²)
    # ------------------------------------------------------
    sapwood_df = pd.read_csv(f"{folder}\\sapwood.csv")
    sapwood_area_vox = sapwood_df["area_opened_AvE"].mean()
    sapwood_area_mm2 = sapwood_area_vox * (voxel_size_mm**2)

    # ------------------------------------------------------
    # 2. fits.csv → radii and z presence
    # ------------------------------------------------------
    fits = pd.read_csv(f"{folder}\\fits\\fits.csv")

    print("\n====== DEBUG FOR SAMPLE:", sample_name, "======")
    print("Raw radius values (vox):", fits["radius"].describe())

    # parse z_present
    def parse_z(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except:
            return []
    fits["z_present"] = fits["z_present"].apply(parse_z)

    fits["z_len"] = fits["z_present"].apply(len)
    print("z_len stats:", fits["z_len"].describe())

    # lengths
    L_m = fits["z_len"] * voxel_size_m
    print("L_m (m):", L_m.describe())

    # radii (m)
    r_m = fits["radius"].astype(float).values * voxel_size_m
    print("r_m (m): min", r_m.min(), " max", r_m.max())

    # compute P50
    P50 = alpha * (r_m ** (-k))
    print("P50 stats:", pd.Series(P50).describe())

    # check if exploding
    if np.nanmax(P50) > 100:
        print("⚠ WARNING: P50 extremely large → cavitation probability collapses")

    # logistic mid values
    midP = 1.5  # around center of your 0-3 MPa range
    Pi_test = 1/(1+np.exp(s*(P50 - midP)))
    print("Pi(midP) stats:", pd.Series(Pi_test).describe())

    # check Ki using correct equation: π r^4 / (8η L)
    Ki = np.zeros_like(r_m)
    mask = (L_m > 0)
    Ki[mask] = np.pi * r_m[mask]**4 / (8*eta*L_m[mask])
    print("Ki stats:", pd.Series(Ki).describe())

    if Ki.sum() == 0:
        print("❌ ALL Ki = 0 → PLC will always be zero")
    if np.nanmax(Pi_test) == 0:
        print("❌ ALL Pi = 0 → PLC will always be zero")

    fits["z_present"] = fits["z_present"].apply(ast.literal_eval)

    fits["z_len"] = fits["z_present"].apply(len)
    fits["L_m"] = fits["z_len"] * voxel_size_m   # length in meters

    # Convert radii (currently voxel units) → meters
    r_vox = fits["radius"].values
    r_m = r_vox * voxel_size_m
    fits["r_m"] = r_m

    # ------------------------------------------------------
    # 3. Conductivity per vessel Ki
    #    Ki = (π r^4 / (8 η)) * L
    # ------------------------------------------------------
    Ki = np.pi * (r_m**4) / (8 * eta) * fits["L_m"]
    fits["Ki"] = Ki

    # ------------------------------------------------------
    # 4. Logistic parameters P50_i = α r^{-k}
    # ------------------------------------------------------
    fits["P50"] = alpha * (r_m**(-k))

    P50 = fits["P50"].values[None, :]
    Ki_col = Ki[None, :]

    P_grid = P[:, None]

    # Logistic cavitation probability
    Pi = 1 / (1 + np.exp(s * (P50 - P_grid)))

    # PLC(P) = Σ(K_i * Pi) / Σ(K_i)
    PLC = 100 * (Pi * Ki_col).sum(axis=1) / Ki.sum()

    # ------------------------------------------------------
    # 5. Extra metrics per sample
    # ------------------------------------------------------

    # mean diameter
    mean_diameter_mm = (2 * r_vox * voxel_size_mm).mean()

    # vessel density = (# vessels) / (sapwood area)
    vessel_density_mm2 = len(fits) / sapwood_area_mm2

    # lumen fraction
    lumen_area_mm2 = np.pi * ((r_vox * voxel_size_mm)**2).sum()
    lumen_fraction = lumen_area_mm2 / sapwood_area_mm2

    # Alternate K_max: use mean radius per z
    # (1) Build z → radii map
    z_radii = {}
    for rp in fits["z_present"]:
        # rp is list of slices where that vessel is present
        pass

    # Flatten all z-slices across vessels
    all_z = sorted({z for lst in fits["z_present"] for z in lst})
    z_mean_r = []

    for z in all_z:
        # collect all vessels present at slice z
        mask = fits["z_present"].apply(lambda L: z in L)
        r_z = fits.loc[mask, "r_m"].values
        if len(r_z) > 0:
            z_mean_r.append(r_z.mean())
        else:
            z_mean_r.append(np.nan)

    z_mean_r = np.array(z_mean_r)
    z_mean_r = z_mean_r[~np.isnan(z_mean_r)]

    # (2) K for each z from mean radius
    Kz = np.pi * (z_mean_r**4) / (8 * eta)

    # (3) average over z then over samples
    K_meanradius = Kz.mean()

    # ------------------------------------------------------
    # 6. Package sample results
    # ------------------------------------------------------
    plc_df = pd.DataFrame({
        "Pressure_MPa": P,
        "PLC_percent": PLC,
        "sample": sample_name,
        "tube_ID": tube_id,
        "biome_site": biome,
        "sapwood_area_mm2": sapwood_area_mm2,
        "K_biome_sample_sumKi": Ki.sum(),
        "mean_diameter_mm": mean_diameter_mm,
        "vessel_density_per_mm2": vessel_density_mm2,
        "lumen_fraction": lumen_fraction,
        "K_meanradius": K_meanradius
    })

    fits["sample"] = sample_name
    fits["biome_site"] = biome

    return plc_df, fits

import pandas as pd
import numpy as np
import ast
from scipy.special import expit  # stable sigmoid

from scipy.special import expit

def process_sample_working_fits(sample_name, tube_id, biome):
    folder = f"{root}\\{sample_name}"

    # ------------------------------------------------------
    # 1. Sapwood.csv
    # ------------------------------------------------------
    try:
        sapwood_df = pd.read_csv(f"{folder}\\sapwood.csv")
        sapwood_area_vox = sapwood_df["area_opened_AvE"].mean()
        sapwood_area_mm2 = sapwood_area_vox * (voxel_size_mm**2)
    except:
        return None, None

    # ------------------------------------------------------
    # 2. fits.csv
    # ------------------------------------------------------
    try:
        fits = pd.read_csv(f"{folder}\\fits\\fits.csv")
    except:
        return None, None

    # Proper parsing of z_present
    def parse_z(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.startswith("["):
            try:
                return ast.literal_eval(x)
            except:
                return []
        return []

    fits["z_present"] = fits["z_present"].apply(parse_z)

    # Length (m)
    fits["z_len"] = fits["z_present"].apply(len)
    fits["L_m"] = fits["z_len"] * voxel_size_m

    # Radii (m)
    r_vox = fits["radius"].astype(float).values
    r_m = r_vox * voxel_size_m
    fits["r_m"] = r_m

    # ------------------------------------------------------
    # 3. Hydraulic conductance per vessel
    #    K_i = π r^4 / (8 η L)
    # ------------------------------------------------------
    L = fits["L_m"].values
    Ki = np.zeros_like(r_m)

    # Safe division (avoid zero-length)
    mask = L > 0
    Ki[mask] = np.pi * (r_m[mask]**4) / (8 * eta * L[mask])
    fits["Ki"] = Ki

    # ------------------------------------------------------
    # 4. P50 = α r^{-k}
    # ------------------------------------------------------
    P50 = alpha * (r_m ** (-k))

    # Pressure curve
    P_grid = P[:, None]  # broadcast

    # Cavitation probability
    Pi = expit(-(s * (P50[None, :] - P_grid)))

    # ------------------------------------------------------
    # 5. PLC
    # ------------------------------------------------------
    if Ki.sum() == 0:
        PLC = np.full_like(P, np.nan)
    else:
        PLC = 100 * np.sum(Pi * Ki[None, :], axis=1) / Ki.sum()

    # ------------------------------------------------------
    # 6. Metrics
    # ------------------------------------------------------
    mean_diameter_mm = (2 * r_vox * voxel_size_mm).mean()
    vessel_density_mm2 = len(fits) / sapwood_area_mm2
    lumen_area_mm2 = np.pi * ((r_vox * voxel_size_mm)**2).sum()
    lumen_fraction = lumen_area_mm2 / sapwood_area_mm2

    # mean radius per z (fix)
    all_z = sorted({z for lst in fits["z_present"] for z in lst})
    z_mean_r = []
    for z in all_z:
        mask = fits["z_present"].apply(lambda L: z in L)
        r_z = fits.loc[mask, "r_m"].values
        if len(r_z) > 0:
            z_mean_r.append(r_z.mean())
    z_mean_r = np.array(z_mean_r)
    Kz = np.pi * (z_mean_r**4) / (8 * eta)  # unit conductance per length
    K_meanradius = Kz.mean() if len(Kz) > 0 else np.nan

    # ------------------------------------------------------
    # 7. Output
    # ------------------------------------------------------
    plc_df = pd.DataFrame({
        "Pressure_MPa": P,
        "PLC_percent": PLC,
        "sample": sample_name,
        "tube_ID": tube_id,
        "biome_site": biome,
        "sapwood_area_mm2": sapwood_area_mm2,
        "sum_Ki": Ki.sum(),
        "mean_diameter_mm": mean_diameter_mm,
        "vessel_density_mm2": vessel_density_mm2,
        "lumen_fraction": lumen_fraction,
        "K_meanradius": K_meanradius
    })

    fits["sample"] = sample_name
    fits["biome_site"] = biome

    return plc_df, fits



### --------------------------------------------------------
### MASTER PIPELINE
### --------------------------------------------------------
results = []
all_fits = []

for _, row in meta.iterrows():
    tube_id = row["tube_ID"]
    biome = row["biome_site"]
    sample_name = f"Sample{tube_id}"

    print(f"Processing {sample_name} (Biome = {biome})")

    try:
        plc_df, fit_df = process_sample_working_fits(sample_name, tube_id, biome)
        if not plc_df.empty and not fit_df.empty:  # skip empty samples
            results.append(plc_df)
            all_fits.append(fit_df)
    except FileNotFoundError:
        print(f"Skipping missing {sample_name}")
    except Exception as e:
        print(f"Error in {sample_name}: {e}")

# Concatenate only if there are valid samples
if results:
    results = pd.concat(results, ignore_index=True)
else:
    results = pd.DataFrame()

if all_fits:
    all_fits = pd.concat(all_fits, ignore_index=True)
else:
    all_fits = pd.DataFrame()


# --------------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------------

out_dir = r"C:\Users\hello\Llewellyn\Analysis\Tiltanalysis\Workingfits\Outputs"
os.makedirs(out_dir, exist_ok=True)

results.to_csv(f"{out_dir}\\results_PLC_curves.csv", index=False)
all_fits.to_csv(f"{out_dir}\\all_fits_combined.csv", index=False)

print("Saved:")
print(f" - {out_dir}\\results_PLC_curves.csv")
print(f" - {out_dir}\\all_fits_combined.csv")

# --------------------------------------------------------
# BIOME-NORMALISED PLC CURVES
# --------------------------------------------------------

if not all_fits.empty:
    biome_groups = all_fits.groupby("biome_site")
    biome_plc_list = []

    for biome, dfb in biome_groups:
        print(f"Biome-normalising: {biome}")

        # flatten vessel list into rows of (r_m, Ki, z_norm)
        rows = []
        for _, row in dfb.iterrows():
            zlist = row["z_present"]
            if len(zlist) < 2:
                continue

            zmin, zmax = min(zlist), max(zlist)
            span = (zmax - zmin)
            if span <= 0:
                continue

            # normalise z for this vessel
            z_norm = [(z - zmin)/span for z in zlist]

            # same radius + Ki for all z-slices
            r_m = row["r_m"]
            Ki = row["Ki"]

            for zn in z_norm:
                rows.append((r_m, Ki, zn))

        if len(rows) == 0:
            print(f"No vessels with z-span in biome {biome}")
            continue

        r_arr = np.array([r for r, K, z in rows])
        Ki_arr = np.array([K for r, K, z in rows])
        # z_norm_arr = np.array([z for r, K, z in rows])  # not needed for PLC calculation

        # biome-level P50 for each z-slice
        P50 = alpha * (r_arr ** (-k))

        # compute biome PLC
        P_grid = P[:, None]
        Pi = expit(-s * (P50[None, :] - P_grid))
        PLC_biome = 100 * np.sum(Pi * Ki_arr[None, :], axis=1) / Ki_arr.sum()

        biome_plc_df = pd.DataFrame({
            "Pressure_MPa": P,
            "PLC_percent": PLC_biome,
            "biome_site": biome,
            "N_vessel_slices": len(rows)
        })

        biome_plc_list.append(biome_plc_df)

    if biome_plc_list:
        biome_plc_df_all = pd.concat(biome_plc_list, ignore_index=True)
        biome_plc_df_all.to_csv(f"{out_dir}\\biome_normalised_PLC_curves.csv", index=False)
        print(f"Saved {out_dir}\\biome_normalised_PLC_curves.csv")

