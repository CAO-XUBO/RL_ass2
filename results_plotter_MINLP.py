import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from distinctipy import distinctipy
import matplotlib.colors as mcolors
from results_plotter_subsets import plot_panel

# Load in MINLP data
allocations_df = pd.read_csv("results\\MINLP_results\\MINLP_robot_assignments.csv")
stations_df = pd.read_csv("results\\MINLP_results\\MINLP_subset_results_summary.csv")

# Store data in a dictionary
data = {}
method = "MINLP"

# Clean the dataframes
stations_df = stations_df[["Density_Type", "Station_Index", "X_Coord", "Y_Coord"]].copy()
stations_df.rename(columns = {"Station_Index": "station_id", "X_Coord": "longitude", "Y_Coord": "latitude"}, inplace=True)
allocations_df = allocations_df[["Density_Type", "Robot_Index", "Robot_X", "Robot_Y", "Assigned_Station_Index"]].copy()
allocations_df.rename(columns = {"Robot_X": "longitude", "Robot_Y": "latitude", "Assigned_Station_Index": "station_id"}, inplace=True)

# Split dataframes by subset densities
for density in stations_df["Density_Type"].unique():
    stations = stations_df[stations_df["Density_Type"] == density].copy()
    # Assign a unique color to each station
    random.seed(42)
    colors = distinctipy.get_colors(len(stations))
    stations.loc[:, "color"] = [mcolors.to_hex(c) for c in colors]

    # Extract the corresponding robots in this subset
    allocations = allocations_df[allocations_df["Density_Type"] == density].copy()
    
    # Store the processed data
    data[(method, density)] = (allocations, stations)

# Draw the 1*3 plot for MINLP
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
for i, density in enumerate(["High", "Median", "Low"]):
        allocations_df, stations_df = data[(method, density)]
        show_legend = True
        draw_labels = True
        plot_panel(allocations_df, stations_df, density, ax = axes[i], show_legend = show_legend, draw_labels = draw_labels)

plt.suptitle(f"Robot Allocations and Station Locations ({method} Method)", fontsize = 18)
plt.tight_layout(rect = [0, 0.03, 1, 0.95])
plt.savefig(f"Diagrams/{method}_method_subsets.png", dpi = 150, bbox_inches = "tight")
    