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

def stochastic_plotter(stochastic_df, scenario_name):
    """
    Draw both robots and stations' locations on a map.
    """

    # Draw robots and stations' locations on a map
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())
    # Small extension to include all points
    lon_min, lon_max = all_robots["longitude"].min() - 5, all_robots["longitude"].max() + 5
    lat_min, lat_max = all_robots["latitude"].min() - 5, all_robots["latitude"].max() + 5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs = ccrs.PlateCarree())

    # Customize
    ax.set_aspect("auto")
    ax.add_feature(cfeature.LAND, facecolor = "white", edgecolor = "black", linewidth = 0.5)
    ax.add_feature(cfeature.OCEAN, facecolor = "lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth = 0.5)

    # Plot station as a blue triangle
    ax.scatter(
        x = stochastic_df["longitude"], 
        y = stochastic_df["latitude"],
        c = "blue",
        marker = "^",
        s = 80,
        alpha = 0.5,
        transform = ccrs.PlateCarree(),
        label = f"Stations (n={len(stochastic_df)})"
    )

    # Plot robot as a dimgray point
    ax.scatter(
        x = all_robots["longitude"], 
        y = all_robots["latitude"],
        c = "dimgray",
        s = 8,
        alpha = 0.8,
        transform = ccrs.PlateCarree(),
        label = f"Robots (n={len(all_robots)})"
    )

    # Labels and legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Robot and Stations' Location in Antarctica")
    ax.legend(loc = "upper right")
    ax.gridlines(draw_labels = True, linewidth = 0.5, color = "gray", alpha = 0.5, linestyle = "--")

    # Magnify the region: 73W ~ 53W, 71S ~ 64S
    ax_inset = fig.add_axes([0.06, 0.57, 0.15, 0.3], projection = ccrs.PlateCarree())
    ax_inset.set_box_aspect(1)
    ax_inset.set_extent([-73, -53, -71, -64], crs = ccrs.PlateCarree())
    ax_inset.add_feature(cfeature.LAND, facecolor = "white", edgecolor = "black", linewidth = 0.5)
    ax_inset.add_feature(cfeature.OCEAN, facecolor = "lightblue")
    ax_inset.add_feature(cfeature.COASTLINE, linewidth = 0.5)
    
    # Draw the magnified area with same data
    ax_inset.scatter(
        x = all_robots["longitude"],
        y = all_robots["latitude"],
        c = "dimgray",
        s = 4,
        edgecolor = "black",
        linewidth = 0.3,
        transform = ccrs.PlateCarree(),
        zorder = 3
    )

    for idx, station_row in stochastic_df.iterrows():
        station_lon = station_row["longitude"]
        station_lat = station_row["latitude"]

        ax_inset.scatter(
            x = station_lon,
            y = station_lat,
            c = "blue",
            marker = "^",
            s = 40,
            alpha = 0.5,
            transform = ccrs.PlateCarree(),
            zorder = 2
        )

    # Save the diagram as PNG
    output_path = Path("Diagrams") / f"stochastic_{scenario_name}_map.png"
    output_path.parent.mkdir(parents = True, exist_ok = True)
    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches = "tight")

# Change file path to plot different scenarios
if __name__ == "__main__":
    stochastic_df_dir = "results\\q2_sensitivity\\stations_lambda_0.016.csv"
    stochastic_df = pd.read_csv(stochastic_df_dir)
    scenario_name = Path(stochastic_df_dir).stem

    stochastic_plotter(stochastic_df, scenario_name)
