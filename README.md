# Predicting Salmon Runs in three River Systems

This project aims to model and predict salmon returns across different river systems (Fraser River, Bristol Bay, Columbia River) using a combination of biological, environmental, and oceanographic data.

## Project Description

Salmon runs are complex ecological events influenced by multiple factors across freshwater and marine environments. 
This project integrates data from return tables, oceanographic data and environmental data to predict the number of returning salmon for specific rivers at three river systems.

Key components of the project include:

- **Data Extraction and Merging**: Extract data from different sources (e.g., PANCEA) and merge them to get annual samples for each river (14 rivers in total).
- **Data Analysis**: Visualize data and look for correlations between the differend variables.
- **Machine Learning Models**: Apply different Machine Learning Models (e.g., GBRT) to forecast salmon returns based on (XXX) features.

## Data Sources

Features in Combined_FeatureSet_For_Model.csv for predicting Total_Returns_NextYear:
- Total_Returns and 18 Age Classes: Return tables (by river and system) provided by Angler's Atlas
- Total_Spawners_BroodYear: Data provided by Gottfried Pestal
- Pacea_ALPI_Anomaly, npi_mean_NovMar, oni_mean_DecFeb, mei_mean_AprSep, npgo_mean_DecFeb, ao_mean_DecMar, pdo_mean_DecMar, pdo_mean_MaySep: recieved from [pacea R package](https://github.com/pbs-assess/pacea) (downloaded 06/16/2025), see notebooks/get_pacea_data.R
- sst_aprjul, sst_anom: NOAA Extended Reconstructed SST V5 data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov, see notebooks/get_SST_v5.py
- sss_aprjun, sss_mayaug: (lat_min, lat_max, lon_min, lon_max) BristolBay: (54, 60, -165, -155), ColumbiaRiver: (45, 48, -125, -121), FraserRiver: (48, 51, -126, -121)
	- Fraser River: Departure Bay (PBS, Lat: 49.21, Lon: -123.955) monthly from  [BC Lightstations]{https://open.canada.ca/data/en/dataset/719955f2-bf8e-44f7-bc26-6bd623e82884/resource/0082007f-5f76-4adb-9c7e-f325e4f838c8}
	- Columbia River and Bristol Bay: [Met Office Hadley Centre observation datasets EN.4.2.2-C13]{https://www.metoffice.gov.uk/hadobs/en4/} (Cowley et al. (2013) XBT corrections and Levitus et al (2009) MBT corrections)
	
- Environmental data from the [pacea R package](https://github.com/pbs-assess/pacea), including:
  - Oceanographic variables from BCCM and HOTSSea models.
  - Climatic indices like PDO, NPGO, MEI, etc.
  - NOAA's OISST sea surface temperatures.

## Goals

- Predict salmon runs in 2025 for the [Salmon Prize Project]{https://salmonprize.com/}.

## Tech Stack

- Python (pandas, seaborn, matplotlib, scikit-learn)
- Jupyter Notebooks
- R (used to extract and clean data from the `pacea` package)
- Git & GitHub for version control


## Contact

For questions, contact Julia at `schmidj` on GitHub.
