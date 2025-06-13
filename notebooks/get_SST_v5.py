# %% Extended Reconstructed Sea Surface Temperature
import xarray as xr
import pandas as pd
import os
from glob import glob
import numpy as np

# Columbia River Mouth (Lat: 45.5°N to 47.0°N, Long: 235.0°E to 236.5°E (which corresponds to 125.0°W to 123.5°W in a 0–360° system))
# Fraser River Mouth (Lat: 48.0°N to 50.0°N, Long: 234.0°E to 238.0°E (124.0°W to 123.0°W))
# Bristol Bay (Lat: 57.0°N to 59.0°N, Long: 198.0°E to 203.0°E (162.0°W to 157.0°W))

regions = {
    "ColumbiaRiver": {"lat": (45.5, 47.0), "lon": (235.0, 236.5)},
    "FraserRiver": {"lat": (48.0, 50.0), "lon": (234.0, 238.0)},
    "BristolBay": {"lat": (57.0, 59.0), "lon": (198.0, 203.0)},
}

files = sorted(glob('../data/ersst.v5/ersst.v5.*.nc'))

data = []

# First, collect SSTs for climatology calculation (April-July, multiple years)
clim_values = {region: [] for region in regions}
clim_years = []

for f in files:
    filename = os.path.basename(f)
    try:
        yyyymm = filename.split('.')[-2]
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])
    except:
        continue

    if month not in [4, 5, 6, 7]:
        continue

    ds = xr.open_dataset(f)

    for region_name, bounds in regions.items():
        lat_slice = slice(*bounds["lat"])
        lon_slice = slice(*bounds["lon"])

        sst_val = ds['sst'].sel(lat=lat_slice, lon=lon_slice).mean().item()

        clim_values[region_name].append(sst_val)
    clim_years.append(year)

# Compute climatological mean for each region (mean over all years/months in April–July)
climatology = {region: np.mean(vals) for region, vals in clim_values.items()}

# Now collect final data with anomalies = SST - climatology
for f in files:
    filename = os.path.basename(f)
    try:
        yyyymm = filename.split('.')[-2]
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])
    except:
        continue

    if month not in [4, 5, 6, 7]:
        continue

    ds = xr.open_dataset(f)

    for region_name, bounds in regions.items():
        lat_slice = slice(*bounds["lat"])
        lon_slice = slice(*bounds["lon"])

        sst_val = ds['sst'].sel(lat=lat_slice, lon=lon_slice).mean().item()
        anom_val = sst_val - climatology[region_name]

        data.append({
            "region": region_name,
            "year": year,
            "month": month,
            "sst_aprjul": sst_val,
            "sst_anom": anom_val
        })

df = pd.DataFrame(data)

# Compute April-July means per year, per region
summer_df = (
    df[df['month'].isin([4, 5, 6, 7])]
    .groupby(['region', 'year'])[['sst_aprjul', 'sst_anom']]
    .mean()
    .reset_index()
)

pivot_df = summer_df.pivot(index='year', columns='region', values=['sst_aprjul', 'sst_anom'])
pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

print(pivot_df.head())

pivot_df.to_csv("../data/sst_april_july_by_region.csv")


# %%
