#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Construct relative paths
data_dir = os.path.join("..", "data", "Brood_Return_First_Year_At_Sea_Tables")
brood_path = os.path.join(data_dir, "Combined_Brood_Bristol_Columbia_Fraser.csv")
return_path = os.path.join(data_dir, "Combined_Return_Bristol_Columbia_Fraser.csv")

# Load data
brood_df = pd.read_csv(brood_path)
return_df = pd.read_csv(return_path)

#%%
# Function to explore a dataset
def explore_by_system(df, name, year_col, total_col):
    print(f"\n📊 Exploring {name} Table by System:\n")
    
    systems = df['System'].unique()
    
    for system in systems:
        print(f"\n📂 System: {system}")
        df_sys = df[df['System'] == system]

        print("Head:\n", df_sys.head())
        print("\nInfo:")
        print(df_sys.info())
        
        print("\nMissing Values:\n", df_sys.isnull().sum())
        print("\nSummary Statistics:\n", df_sys.describe())

        # Line plot of total returns/recruits
        if year_col in df_sys.columns and total_col in df_sys.columns:
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=df_sys, x=year_col, y=total_col)
            plt.title(f"{system} - {name}: {total_col} Over Time")
            plt.ylabel(total_col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Age composition heatmap
        age_cols = [col for col in df_sys.columns if col.startswith("AgeClass_")]

        # Keep only age columns that have at least one non-zero and non-NaN value
        valid_age_cols = [
            col for col in age_cols
            if not ((df_sys[col].isna()) | (df_sys[col] == 0)).all()
        ]

        if valid_age_cols:
            age_df = df_sys[[year_col] + valid_age_cols].dropna(how="all", subset=valid_age_cols).set_index(year_col)
            age_df = age_df.fillna(0)
            if not age_df.empty and age_df.to_numpy().sum() > 0:
                plt.figure(figsize=(12, 6))
                sns.heatmap(age_df.T, cmap="YlGnBu", cbar_kws={'label': 'Count'})
                plt.title(f"{system} - {name}: Age Composition by Year")
                plt.xlabel("Year")
                plt.ylabel("Age Class")
                plt.tight_layout()
                plt.show()
            else:
                print(f"⚠️ Skipping heatmap for {system} - age data is empty or all zeros.")
        else:
            print(f"⚠️ Skipping heatmap for {system} - no valid age columns.")


#%%
# Run exploration
explore_by_system(brood_df, "Brood", year_col="BroodYear", total_col="Total_Recruits")
explore_by_system(return_df, "Return", year_col="ReturnYear", total_col="Total_Returns")

# %%
# Cumulative Recruits vs Returns by River and System
sns.set(style="whitegrid")

brood_totals = (
    brood_df.groupby(["System", "River", "BroodYear"])["Total_Recruits"]
    .sum()
    .reset_index()
    .rename(columns={"BroodYear": "Year", "Total_Recruits": "Total"})
)
return_totals = (
    return_df.groupby(["System", "River", "ReturnYear"])["Total_Returns"]
    .sum()
    .reset_index()
    .rename(columns={"ReturnYear": "Year", "Total_Returns": "Total"})
)

brood_totals["Cumulative"] = brood_totals.groupby(["System", "River"])["Total"].cumsum()
brood_totals["Type"] = "Recruits"

return_totals["Cumulative"] = return_totals.groupby(["System", "River"])["Total"].cumsum()
return_totals["Type"] = "Returns"

combined = pd.concat([
    brood_totals[["System", "River", "Year", "Cumulative", "Type"]],
    return_totals[["System", "River", "Year", "Cumulative", "Type"]]
])

# Assign a color per river
unique_rivers = combined["River"].unique()
palette = dict(zip(unique_rivers, sns.color_palette("tab10", n_colors=len(unique_rivers))))

# Create the FacetGrid
g = sns.FacetGrid(
    combined,
    col="System",
    col_wrap=1,
    height=4,
    aspect=2,
    sharey=False
)

# Custom plotting function
def plot_lines(data, **kwargs):
    for river in data["River"].unique():
        subset = data[data["River"] == river]
        color = palette[river]
        
        # Recruits = solid line
        recruits = subset[subset["Type"] == "Recruits"]
        plt.plot(recruits["Year"], recruits["Cumulative"], label=f"{river} - Recruits", linestyle="solid", color=color)
        
        # Returns = dashed line
        returns = subset[subset["Type"] == "Returns"]
        plt.plot(returns["Year"], returns["Cumulative"], label=f"{river} - Returns", linestyle="dashed", color=color)

# Map plot function
g.map_dataframe(plot_lines)
g.set_axis_labels("Year", "Cumulative Fish Count")
g.set_titles("{col_name}")
g.add_legend(title="River - Line Type")
plt.tight_layout()
plt.show()
# %%
# Cohort reconstruction using age-at-return data
# Melt age composition data to long format
age_cols = [col for col in return_df.columns if col.startswith("AgeClass_")]
age_long = brood_df.melt(
    id_vars=["BroodYear", "System", "River"],
    value_vars=age_cols,
    var_name="AgeClass",
    value_name="NumFish"
)
# Extract years spent from AgeClass column
age_long[["FreshwaterYears", "SaltwaterYears"]] = age_long["AgeClass"].str.extract(r"AgeClass_(\d)\.(\d)").astype(float)
# Calculate ReturnYear = BroodYear + freshwater + saltwater
age_long["ReturnYear"] = age_long["BroodYear"] + age_long["FreshwaterYears"] + age_long["SaltwaterYears"]
# Aggregate returns by BroodYear and ReturnYear
cohort_returns = age_long.groupby(["BroodYear", "ReturnYear", "System", "River"])["NumFish"].sum().reset_index()


# %%
# Plot some data