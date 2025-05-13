#%%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Paths to .csv files
data_dir = os.path.join("..", "data")
#fish_features = os.path.join(data_dir, "Combined_Brood_Bristol_Columbia_Fraser.csv")
fish_features = os.path.join(data_dir, "Brood_Return_First_Year_At_Sea_Tables", "Combined_Return_Bristol_Columbia_Fraser.csv")
pancea_features = os.path.join(data_dir, "GENERATED_pacea_series_annual.csv")

# Load feature sheets
fish_df = pd.read_csv(fish_features)
pancea_df = pd.read_csv(pancea_features)

#%% Prepare fish_df
# Rename ReturnYear for clarity
fish_df = fish_df.rename(columns={'ReturnYear': 'Year'})

# Ensure years are sorted before shifting
fish_df = fish_df.sort_values(by=['System', 'River', 'Year'])

# Create prediction target: next year's Total_Returns
fish_df['Total_Returns_NextYear'] = fish_df.groupby(['System', 'River'])['Total_Returns'].shift(-1)

# Merge with climate/ocean features (PANCEA)
combined_df = fish_df.merge(pancea_df, how='left', on='Year')

# Drop rows where target is missing (i.e., last year of each group) and index column
combined_df = combined_df.dropna(subset=['Total_Returns_NextYear'])
combined_df = combined_df.drop(columns=["Unnamed: 0"])

#%% Check for constant columns
constant_cols = [col for col in combined_df.columns if combined_df[col].nunique() <= 1]
if constant_cols:
    print("Columns with only one unique value (consider dropping):")
    print(constant_cols)
else:
    print("No constant columns found.")

#%% Correlation matrix
# Select only numeric columns for correlation
numeric_df = combined_df.select_dtypes(include=[np.number])

# Compute correlation
correlation_matrix = numeric_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(16, 12))

# Generate the heatmap
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})

plt.title("Correlation Heatmap of Numeric Features", fontsize=16)
plt.tight_layout()


# Identify and print highly correlated feature pairs
def find_highly_correlated_features(corr_matrix, threshold=0.9):
    correlated_pairs = []
    to_remove = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                correlated_pairs.append((col1, col2, corr_value))
                to_remove.add(col2)  # Arbitrarily keep col1 and suggest removing col2

    return correlated_pairs, to_remove

high_corr_pairs, features_to_remove = find_highly_correlated_features(correlation_matrix, threshold=0.9)

# Print correlated pairs
print("\nHighly correlated feature pairs (|correlation| > 0.9):")
for col1, col2, corr_val in high_corr_pairs:
    print(f"{col1} <--> {col2} | correlation: {corr_val:.2f}")

# Print features to consider removing
print("\nFeatures to consider removing due to high correlation:")
for feature in sorted(features_to_remove):
    print(f"- {feature}")

#%% Drop highly correlated or redundant columns
columns_to_drop = ['Pacea_NPI_Anomaly', 'Pacea_NPI_Value']
combined_df.drop(columns=columns_to_drop, inplace=True)

print(f"Dropped columns: {columns_to_drop}")

#%% Export to .csv
output_path = os.path.join(data_dir, "Combined_FeatureSet_For_Model.csv")
combined_df.to_csv(output_path, index=False)

num_samples = combined_df.shape[0]
num_features = combined_df.shape[1] - 1  # Exclude target variable if desired


print(f"Combined dataset saved to {output_path} with {num_samples} samples and {num_features} features.")

# %% Save separate files for each system
for system in combined_df['System'].unique():
    system_df = combined_df[combined_df['System'] == system].copy()
    system_df = system_df.drop(columns=["System"])

    # Compute correlation matrix for numeric features only
    numeric_cols = system_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols]
    corr_matrix = system_df[feature_cols].corr()

    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= 0.9:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], round(corr, 3)))

    # Print results
    if high_corr_pairs:
        print(f"Highly correlated features in {system}:")
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"  {feat1} <--> {feat2}: correlation = {corr_val}")
    else:
        print(f"No highly correlated features (|r| ≥ 0.9) found in {system}.")

        
    # Drop specific correlated features for each system
    if system == "Columbia River":
        drop_cols = ['AgeClass_1.2', 'AgeClass_3.3']
        system_df = system_df.drop(columns=[col for col in drop_cols if col in system_df.columns])

    elif system == "Fraser River":
        drop_cols = ['AgeClass_1.2', 'Pacea_ALPI_Anomaly']
        system_df = system_df.drop(columns=[col for col in drop_cols if col in system_df.columns])
    
    # Save cleaned file
    system_path = os.path.join(data_dir, f"{system}_FeatureSet_For_Model.csv")
    system_df.to_csv(system_path, index=False)

    sys_samples = system_df.shape[0]
    sys_features = system_df.shape[1] - 1

    print(f"{system} dataset saved to {system_path} with {sys_samples} samples and {sys_features} features.")

    # %%
