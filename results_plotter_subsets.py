import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from distinctipy import distinctipy
import matplotlib.colors as mcolors


# Load in robots' location data
all_robots = pd.read_csv("processed_data/robot_locations_range.csv")
all_robots.rename(columns={'index': 'robot_id'}, inplace=True) # Change column name

# Folder names for the data of 3 subset
density_folders = {
	"High": "heuristic_deterministic_high",
	"Medium": "heuristic_deterministic_median",
	"Low": "heuristic_deterministic_low",
}

# Methods
methods = ["deterministic", "local_search"]

# Load in data
data = {}

for density, folder in density_folders.items():
    for method in methods:
        allocations_dir = Path("results") / folder / f"allocations_{method}.csv"
        stations_dir = Path("results") / folder / f"stations_{method}.csv"
        allocations_df = pd.read_csv(allocations_dir)
        allocations_df = pd.merge(allocations_df, all_robots, on="robot_id", how="left") # Match robot locations
        stations_df = pd.read_csv(stations_dir)
        # Assign a unique color to each station
        np.random.seed(42)
        colors = distinctipy.get_colors(len(stations_df))
        stations_df["color"] = [mcolors.to_hex(c) for c in colors]
        data[(density, method)] = (allocations_df, stations_df)


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
    for i, density in enumerate(["High", "Medium", "Low"]):
        allocations_df, stations_df = data[(density, m)]
        show_legend = True
        draw_labels = True
        plot_panel(allocations_df, stations_df, density, ax=axes[i], show_legend=show_legend, draw_labels=draw_labels)
    plt.suptitle(f"Robot Allocations and Station Locations ({m.capitalize()} Method)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"Diagrams/{m}_method_subsets.png", dpi = 150, bbox_inches = "tight")
    