import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from distinctipy import distinctipy
import matplotlib.colors as mcolors


# Load in robots' location data
all_robots = pd.read_csv("processed_data/robot_locations_range.csv")
all_robots.rename(columns={'index': 'robot_id'}, inplace=True) # Change column name

# Methods folder
methods = ["deterministic", "local_search"]
method_folder = {
    "deterministic": "heuristic_deterministic",
    "local_search": "local_search_deterministic",
}

# Write subsets data in a dictionary
data = {}

for method in methods:
    path = Path("results") / method_folder[method]
    for subset_folder in path.iterdir():
        if not subset_folder.is_dir():
            continue
        # Extract density from folder name
        density = subset_folder.name.split("_")[-1].lower()
        if density == "full":
            continue  # skip full instance
        density = density.capitalize()

        allocations_path = subset_folder / f"allocations_{method}.csv"
        stations_path = subset_folder / f"stations_{method}.csv"

        if not allocations_path.exists() or not stations_path.exists():
            continue

        allocations_df = pd.read_csv(allocations_path)
        stations_df = pd.read_csv(stations_path)
        # Preprocess data
        # 1) Match robots' location
        allocations_df = pd.merge(allocations_df, all_robots, on = "robot_id", how = "left")
        # 2) Assign unique color to each station
        random.seed(42)
        # Use station_id-sorted mapping to keep colors stable across subsets and reruns.
        stations_df = stations_df.sort_values("station_id").reset_index(drop = True)
        colors = distinctipy.get_colors(len(stations_df))
        stations_df["color"] = [mcolors.to_hex(c) for c in colors]

        # Store the processed data
        data[(method, density)] = (allocations_df, stations_df)


def plot_panel(allocations_df, stations_df, density, ax = None, extent = None, show_legend = True, draw_labels = False):
    """
    Plot panel for each subset and method.
    """

    created_figure = False
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        created_figure = True

    # Use a shared extent
    if extent is None:
        lon_min, lon_max = allocations_df["longitude"].min() - 5, allocations_df["longitude"].max() + 5
        lat_min, lat_max = allocations_df["latitude"].min() - 5, allocations_df["latitude"].max() + 5
        extent = [lon_min, lon_max, lat_min, lat_max]

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Customize the map
    ax.set_aspect("auto")
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # Plot stations and their allocated robots using the same station color.
    for _, station_row in stations_df.iterrows():
        station_id = station_row["station_id"]
        station_lon = station_row["longitude"]
        station_lat = station_row["latitude"]
        station_color = station_row["color"]

        # Find robots allocated to this station
        assigned_robots = allocations_df[allocations_df["station_id"] == station_id]

        # Draw the robot as a circle
        ax.scatter(
            x = assigned_robots["longitude"],
            y = assigned_robots["latitude"],
            c = station_color,
            s = 8,
            transform = ccrs.PlateCarree(),
            zorder = 3,
        )

        # Draw the station as a triangle
        ax.scatter(
            x = station_lon,
            y = station_lat,
            c = station_color,
            marker = "^",
            s = 80,
            alpha = 0.5,
            transform = ccrs.PlateCarree(),
            zorder = 2,
        )

    # Label and legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{density} Subset")
    if show_legend:
        ax.scatter([], [], c = "gray", s = 8, marker = "o", label = f"Robots (n={len(allocations_df)})", transform = ccrs.PlateCarree())
        ax.scatter([], [], c = "gray", s = 80, marker = "^", label = f"Stations (n={len(stations_df)})", transform = ccrs.PlateCarree())
        ax.legend(loc = "upper right", scatterpoints = 1)
    ax.gridlines(draw_labels = draw_labels, linewidth = 0.5, color = "gray", alpha = 0.5, linestyle = "--")

    if created_figure:
        plt.tight_layout()

    return ax


# Draw 1*3 plot for each method 
for m in methods:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    for i, density in enumerate(["High", "Median", "Low"]):
        allocations_df, stations_df = data[(m, density)]
        show_legend = True
        draw_labels = True
        plot_panel(allocations_df, stations_df, density, ax = axes[i], show_legend = show_legend, draw_labels = draw_labels)
    plt.suptitle(f"Robot Allocations and Station Locations ({m.capitalize()} Method)", fontsize = 18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"Diagrams/{m}_method_subsets.png", dpi = 150, bbox_inches = "tight")
    