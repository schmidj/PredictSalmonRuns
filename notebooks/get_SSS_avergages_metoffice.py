#%%
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Set the directory where all .nc files are located
data_dir = "..\data\sss_metoffice\en4_analysis_nc"

# Define region bounding boxes (lat_min, lat_max, lon_min, lon_max)
regions = {
    "BristolBay": (54, 60, -165, -155),
    "ColumbiaRiver": (45, 48, -125, -121),
    "FraserRiver": (48, 51, -126, -121)
}

# Adjust longitudes to 0–360 if needed
def wrap_lon(lon):
    return lon if lon >= 0 else lon + 360

def extract_avg_sss(nc_path, lat_bounds, lon_bounds):
    with Dataset(nc_path) as nc:
        print(nc)
        lats = nc.variables["lat"][:]
        lons = nc.variables["lon"][:]
        sss = nc.variables["salinity"][0, 0, :, :]  # surface level, time dim is length 1

        # Adjust lon bounds
        lon_min, lon_max = map(wrap_lon, lon_bounds)
        lons = np.array([wrap_lon(lon) for lon in lons])

        lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
        lon_mask = (lons >= lon_min) & (lons <= lon_max)

        sss_region = sss[np.ix_(lat_mask, lon_mask)]
        sss_avg = np.nanmean(sss_region)

    return sss_avg

# Main loop
results = []

for year in range(1942, 2025):
    sss_values = {"Year": year}

    for region, (lat_min, lat_max, lon_min, lon_max) in regions.items():
        aprjun_vals = []
        mayaug_vals = []

        for month in range(1, 13):
            if month in [4, 5, 6]:  # April–June
                fname = f"EN.4.2.2.f.analysis.c13.{year}{month:02d}.nc"
                path = os.path.join(data_dir, fname)
                print(path)
                if os.path.exists(path):
                    aprjun_vals.append(extract_avg_sss(path, (lat_min, lat_max), (lon_min, lon_max)))
            if month in [5, 6, 7, 8]:  # May–August
                fname = f"EN.4.2.2.f.analysis.c13.{year}{month:02d}.nc"
                path = os.path.join(data_dir, fname)
                if os.path.exists(path):
                    mayaug_vals.append(extract_avg_sss(path, (lat_min, lat_max), (lon_min, lon_max)))

        sss_values[f"{region}_sss_aprjun"] = np.nanmean(aprjun_vals) if aprjun_vals else np.nan
        sss_values[f"{region}_sss_mayaug"] = np.nanmean(mayaug_vals) if mayaug_vals else np.nan

    results.append(sss_values)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("..\data\sss_aprjun_mayaug_by_region.csv", index=False)
print("Saved to sss_aprjun_mayaug_by_region.csv")
# %%
