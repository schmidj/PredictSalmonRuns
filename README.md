# Predicting Salmon Runs in three River Systems

This project aims to model and predict salmon returns across different river systems (Fraser River, Bristol Bay, Columbia River) using a combination of biological, environmental, and oceanographic data.

## Project Description

Salmon runs are complex ecological events influenced by multiple factors across freshwater and marine environments. 
This project integrates data from brood tables, return tables, and marine entry records to predict the number of returning salmon for specific rivers and systems.

Key components of the project include:

- **Brood Year Analysis**: Tracks the number of juvenile salmon by age class and year.
- **Return Year Analysis**: Captures observed returns, used to validate recruitment predictions.
- **First Year at Sea**: Estimates of marine entry and early marine survival by age class.
- **Environmental Covariates**: Incorporates sea surface temperature, salinity, dissolved oxygen, and large-scale climate indices (e.g., PDO, NPGO, MEI).
- **Machine Learning Models**: Applied to forecast salmon returns based on prior brood years and environmental drivers.

## Data Sources

- Angler's Atlas provided brood and return tables (by river and system).
- Marine entry age-class data.
- Environmental data from the [pacea R package](https://github.com/pbs-assess/pacea), including:
  - Oceanographic variables from BCCM and HOTSSea models.
  - Climatic indices like PDO, NPGO, MEI, etc.
  - NOAA's OISST sea surface temperatures.

## Goals

- Predict future salmon runs to support fisheries management.

## Tech Stack

- Python (pandas, seaborn, matplotlib, scikit-learn)
- Jupyter Notebooks
- R (used to extract and clean data from the `pacea` package)
- Git & GitHub for version control


## Contact

For questions, contact Julia at `schmidj` on GitHub.
