# Bayesian Neural Network for Predicting Recreational Angler Activity

This project uses a Bayesian Neural Network (BNN) to model boat counts at lakes based on environmental and socioeconomic predictors.

## Motivation
Understanding and quantifying uncertainty in human-environment interaction is key in ecological management. Traditional models (GLM, RF, SVM, etc.) performed similarly — a BNN was used to incorporate uncertainty explicitly.

## Highlights
- Deep learning with TensorFlow Probability
- Uncertainty-aware predictions
- Real-world application using angler app data

## Project Structure
angler-bnn-project/
├── data/
│ └── processed/ # Cleaned CSV files
├── notebooks/
│ └── 01_bnn_model.ipynb # Main experiment notebook
├── src/
│ ├── preprocess.py # Data loading & scaling
│ ├── model.py # BNN model definition
│ └── evaluate.py # Plotting with uncertainty
├── README.md
└── requirements.txt

## Folder Overview
- `data/`: Raw + processed data
- `src/`: Scripts for preprocessing, modeling, and evaluation
- `notebooks/`: Training and evaluation notebook
- `figures/`: Prediction plots with uncertainty
- `models/`: (Optional) Saved models

## How to Run
1. Install requirements:
pip install -r requirements.txt

2. Run the notebook:
jupyter notebook notebooks/01_bnn_model.ipynb