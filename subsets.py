import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyproj import Transformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# PARAMETERS AVAILABLE TO BE ADJUSTED
subset_size = 40            # Size of subsets
distance_threshold = 10     # Degree threshold to avoid overlap between subsets




# Load in the robot location data
robots = pd.read_csv("processed_data/robot_locations_range.csv")
lon = robots["longitude"].values
lat = robots["latitude"].values
coords = np.column_stack([lon, lat])

# Use KNN to find the 10 nearest neighbors for each robot
neighbors = NearestNeighbors(n_neighbors = 10 + 1, metric = "euclidean") # +1 to include the robot itself
neighbors.fit(coords)
distances, indices = neighbors.kneighbors(coords)

# Compute the local density for each robot 
# the inverse of the mean distance to the 10 nearest neighbors for each robot
knn_distances = distances[:, 1:]   # Exclude the itself
loc_density = 1 / (knn_distances.mean(axis = 1))
robots["loc_density"] = loc_density


def get_subset_around_center(center_idx, candidates_idx, candidates_coords):
    """
    Find the 50 nearest neighbors of the center point.
    """
    # Fit NearestNeighbors
    knn = NearestNeighbors(n_neighbors = subset_size, metric = "euclidean")
    knn.fit(candidates_coords)
    
    # Local position of the center
    local_idx = candidates_idx.index(center_idx)
    center_coord = candidates_coords[local_idx].reshape(1, -1)
    
    # Local indices of the 50 nearest neighbors
    local_neighbor_idx = knn.kneighbors(center_coord, return_distance = False)[0]
    
    # Convert into original indices
    subset_idx = []
    for i in local_neighbor_idx:
        subset_idx.append(candidates_idx[i])
    subset = robots.iloc[subset_idx].copy()
    
    return subset_idx, subset


def filter_by_distance(center_coords, candidate_idx, threshold):
    """
    Filter candidates to keep only those far enough from all center points.
    """
    filtered_idx = []
    for idx in candidate_idx:
        candidate_coord = coords[idx]
        far_enough = all(np.linalg.norm(candidate_coord - c) >= threshold for c in center_coords)
        if far_enough:
            filtered_idx.append(idx)
    return filtered_idx



# High-density subset
highest_density_idx = robots["loc_density"].idxmax()
highest_coord = coords[highest_density_idx]

candidates_idx = list(robots.index)
candidates_coords = coords[candidates_idx]
high_subset_idx, high_subset = get_subset_around_center(highest_density_idx, candidates_idx, candidates_coords)


# Low-density Subset
candidates_idx = list(set(candidates_idx) - set(high_subset_idx)) # Remove high_subset
candidates_coords = coords[candidates_idx]

far_from_high = filter_by_distance([highest_coord], candidates_idx, distance_threshold)
lowest_density_idx = robots.loc[far_from_high, "loc_density"].idxmin()
lowest_coord = coords[lowest_density_idx]
low_subset_idx, low_subset = get_subset_around_center(lowest_density_idx, candidates_idx, candidates_coords)


# Median-density Subset
candidates_idx = list(set(candidates_idx) - set(low_subset_idx)) # Remove low_subset
candidates_coords = coords[candidates_idx]

far_from_both = filter_by_distance([highest_coord, lowest_coord], candidates_idx, distance_threshold)
median_density = robots["loc_density"].median()
density_diff = (robots.loc[far_from_both, "loc_density"] - median_density).abs()
median_density_idx = density_diff.idxmin()
median_subset_idx, median_subset = get_subset_around_center(median_density_idx, candidates_idx, candidates_coords)


# Output the subsets in CSV
high_subset["subset"] = "high"
low_subset["subset"] = "low"
median_subset["subset"] = "median"

#output_cols = ["index", "longitude", "latitude", "range", "subset"]
all_subsets = pd.concat([high_subset, low_subset, median_subset], ignore_index = True)
all_subsets = all_subsets.drop(columns = ["loc_density"])
all_subsets.to_csv("processed_data/robot_subsets.csv", index = False)


# Draw the subsets in a map
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())
# Small extension to include all points
lon_min, lon_max = robots["longitude"].min() - 5, robots["longitude"].max() + 5
lat_min, lat_max = robots["latitude"].min() - 5, robots["latitude"].max() + 5
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs = ccrs.PlateCarree())

# Customize
ax.set_aspect("auto")
ax.add_feature(cfeature.LAND, facecolor = "white", edgecolor = "black", linewidth = 0.5)
ax.add_feature(cfeature.OCEAN, facecolor = "lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth = 0.5)


# Plot high-density subset as red points
ax.scatter(
    x = high_subset["longitude"], 
    y = high_subset["latitude"],
    c = "red",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"High density (n={len(high_subset)})"
)


# Plot median-density subset as yellow points
ax.scatter(
    x = median_subset["longitude"], 
    y = median_subset["latitude"],
    c = "gold",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"Median density (n={len(median_subset)})"
)


# Plot low-density subset as green points
ax.scatter(
    x = low_subset["longitude"], 
    y = low_subset["latitude"],
    c = "green",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"Low density (n={len(low_subset)})"
)

# Plot all other robots as gray points
all_subset_idx = high_subset_idx + low_subset_idx + median_subset_idx
other_idx = []
for i in robots.index:
    if i not in all_subset_idx:
        other_idx.append(i)
other_robots = robots.iloc[other_idx]

ax.scatter(
    x = other_robots["longitude"], 
    y = other_robots["latitude"],
    c = "gray",
    s = 5,
    alpha = 0.5,
    transform = ccrs.PlateCarree(),
    label = f"Other robots (n={len(other_robots)})"
)


ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Robot Locations in Antarctica")
ax.legend(loc = "upper right")
ax.gridlines(draw_labels = True, linewidth = 0.5, color = "gray", alpha = 0.5, linestyle = "--")


# Save the diagram as PNG
plt.tight_layout()
plt.savefig("Diagrams/robot_subset_map.png", dpi = 150, bbox_inches = "tight")




