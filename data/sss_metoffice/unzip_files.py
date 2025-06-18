# %%
import zipfile
import os

zip_dir = "C:/Users/julia/Documents/AnglersProject/SalmonRun/SalmonRun_Project/data/sss_metoffice/en4_analysis_zips"
out_dir = "C:/Users/julia/Documents/AnglersProject/SalmonRun/SalmonRun_Project/data/sss_metoffice/en4_analysis_nc"

os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(zip_dir):
    if fname.endswith(".zip"):
        zip_path = os.path.join(zip_dir, fname)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(out_dir)

print("All .zip files extracted to:", out_dir)

# %%
