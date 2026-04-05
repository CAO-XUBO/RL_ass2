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

def data_preprocess(stations_dir, allocations_dir):
    """
    Preprocess both station location and robot allocation data:
    1) Assign different colors to different station;
    2) Merge robot location data with allocation data.
    """
    # Read data
    stations_df = pd.read_csv(stations_dir)
    allocations_df = pd.read_csv(allocations_dir)

    # Assign a unique color to each station
    np.random.seed(42)
    colors = distinctipy.get_colors(len(stations_df), random_seed=42)
    stations_df["color"] = [mcolors.to_hex(c) for c in colors]
    # Merge robot location data with allocation data
    robot_df = pd.merge(allocations_df, all_robots, on="robot_id", how="left")

    return stations_df, robot_df

def plot_results(stations_df, robot_df, output_path):
    """
    Plot the station locations and robot allocations on a map:
    1) All robots will be plotted as dimgray points;
    2) All stations will be plotted as different colored triangles;
    3) Robots allocated to the same station will be plotted in the same color as that station.
    """
    # Draw the subsets in a map
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())
    # Small extension to include all points
    lon_min, lon_max = robot_df["longitude"].min() - 5, robot_df["longitude"].max() + 5
    lat_min, lat_max = robot_df["latitude"].min() - 5, robot_df["latitude"].max() + 5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs = ccrs.PlateCarree())

    # Customize the map
    ax.set_aspect("auto")
    ax.add_feature(cfeature.LAND, facecolor = "white", edgecolor = "black", linewidth = 0.5)
    ax.add_feature(cfeature.OCEAN, facecolor = "lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth = 0.5)

    # Plot stations and their allocated robots using the same station color.
    for idx, station_row in stations_df.iterrows():
        station_id = station_row["station_id"]
        station_lon = station_row["longitude"]
        station_lat = station_row["latitude"]
        station_color = station_row["color"]
        
        # Find robots allocated to this station
        assigned_robots = robot_df[robot_df["station_id"] == station_id]
        
        # Draw the robot as a circle
        ax.scatter(
            x = assigned_robots["longitude"],
            y = assigned_robots["latitude"],
            c = station_color,
            s = 8,
            #edgecolor = "black",
            #linewidth = 0.5,
            transform = ccrs.PlateCarree(),
            zorder = 3
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
            zorder = 2
        )
    
    # Label and legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station Locations and Robot Allocations in Antarctica")
    legend_robot = ax.scatter([], [], c = "gray", s = 8, marker="o", label = f"Robots (n={len(robot_df)})", transform = ccrs.PlateCarree())
    legend_station = ax.scatter([], [], c = "gray", s = 80, marker="^", label = f"Stations (n={len(stations_df)})", transform = ccrs.PlateCarree())
    ax.legend(loc = "upper right", scatterpoints = 1)
    ax.gridlines(draw_labels = True, linewidth = 0.5, color = "gray", alpha = 0.5, linestyle = "--")

    # Magnify the region: 73W ~ 53W, 71S ~ 64S
    ax_inset = fig.add_axes([0.06, 0.57, 0.15, 0.3], projection = ccrs.PlateCarree())
    ax_inset.set_box_aspect(1)
    ax_inset.set_extent([-73, -53, -71, -64], crs = ccrs.PlateCarree())
    ax_inset.add_feature(cfeature.LAND, facecolor = "white", edgecolor = "black", linewidth = 0.5)
    ax_inset.add_feature(cfeature.OCEAN, facecolor = "lightblue")
    ax_inset.add_feature(cfeature.COASTLINE, linewidth = 0.5)
    
    # Draw the magnified area with same data
    for idx, station_row in stations_df.iterrows():
        station_id = station_row["station_id"]
        station_lon = station_row["longitude"]
        station_lat = station_row["latitude"]
        station_color = station_row["color"]
        
        assigned_robots = robot_df[robot_df["station_id"] == station_id]
        ax_inset.scatter(
            x = assigned_robots["longitude"],
            y = assigned_robots["latitude"],
            c = station_color,
            s = 4,
            edgecolor = "black",
            linewidth = 0.3,
            transform = ccrs.PlateCarree(),
            zorder = 3
        )
        ax_inset.scatter(
            x = station_lon,
            y = station_lat,
            c = station_color,
            marker = "^",
            s = 40,
            alpha = 0.8,
            transform = ccrs.PlateCarree(),
            zorder = 2
        )
    

    # Save the plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches = "tight")
    plt.close(fig)


def generate_maps_for_folder(results_folder):
    """
    For each pair of robot allocation and station location data,
    draw the corresponding map and save it in the same folder.
    """
    results_path = Path(results_folder)
    if not results_path.exists() or not results_path.is_dir():
        raise FileNotFoundError(f"Invalid result folder: {results_folder}")

    for alloc_file in results_path.glob("allocations_*.csv"):
        method_name = alloc_file.stem.replace("allocations_", "", 1)
        station_file = results_path / f"stations_{method_name}.csv"

        if not station_file.exists():
            print(f"Skip {method_name}: missing {station_file.name}")
            continue

        stations_df, robot_df = data_preprocess(station_file, alloc_file)
        output_file = results_path / f"{method_name}_full_map.png"
        plot_results(stations_df, robot_df, output_file)


if __name__ == "__main__":
    target_folder = r"results\\heuristic_deterministic_full"
    generate_maps_for_folder(target_folder)









