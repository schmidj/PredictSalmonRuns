import matplotlib.pyplot as plt

def plot_predictions_by_river(timeline_df):
    """
    Plots predicted vs actual values for each river over time.

    Args:
        timeline_df (pd.DataFrame): DataFrame with 'River_Name', 'Year', 'Predicted', and 'Actual' columns.
    """
    rivers = timeline_df["River_Name"].unique()
    num_rivers = len(rivers)

    # Set up a grid of subplots
    fig, axes = plt.subplots(nrows=num_rivers, ncols=1, figsize=(10, 4 * num_rivers), sharex=True)

    if num_rivers == 1:
        axes = [axes]

    for ax, river in zip(axes, rivers):
        river_df = timeline_df[timeline_df["River_Name"] == river].sort_values("Year")
        ax.plot(river_df["Year"], river_df["Actual"], label="Actual", marker='o')
        ax.plot(river_df["Year"], river_df["Predicted"], label="Predicted", marker='x')
        ax.set_title(f"River: {river}")
        ax.set_ylabel("Total_Returns_NextYear")
        ax.legend()
        ax.grid(True)

    plt.xlabel("Year")
    plt.tight_layout()
    plt.show()
    
def plot_actual_vs_predicted(results, title="Model: Actual vs. Predicted Total Returns (Next Year)"):
    """
    Plots actual vs. predicted values from a results dictionary.

    Args:
        results (dict): Dictionary with keys 'Actual' and 'Predicted'.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results["Actual"], label="Actual", marker='o')
    plt.plot(results["Predicted"], label="Predicted", marker='x')
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Total Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()