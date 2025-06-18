# %%
import requests
import os

base_url = "http://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/EN.4.2.2/"
start_year = 1950
end_year = 1954
output_folder = "en4_analysis_zips"

os.makedirs(output_folder, exist_ok=True)

for year in range(start_year, end_year + 1):
    filename = f"EN.4.2.2.analyses.c13.{year}.zip"
    url = f"{base_url}{filename}"
    out_path = os.path.join(output_folder, filename)

    print(f"Downloading {url}")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200 and 'application/zip' in response.headers.get('Content-Type', ''):
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ Saved: {filename}")
        else:
            print(f"✗ Skipped {year}: status {response.status_code}, content type {response.headers.get('Content-Type')}")
    except Exception as e:
        print(f"✗ Error downloading {year}: {e}")
# %%
