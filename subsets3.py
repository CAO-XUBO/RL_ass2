import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# PARAMETER AVAILABLE TO BE ADJUSTED
subset_size = 50

# Load in the robot location data
robots = pd.read_csv("processed_data/robot_locations_range.csv")

# Randomly sample a subset of robots
subset = robots.sample(subset_size)

# Output the subsets in CSV
subset.to_csv("processed_data/robot_subset3.csv", index=False)

# Draw the results in a map
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

# Plot the subset as red points
ax.scatter(
    x = subset["longitude"], 
    y = subset["latitude"],
    c = "red",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"Subset (n={len(subset)})"
)

# Plot all other robots as gray points
subset_idx = subset.index.tolist()
other_idx = []
for i in robots.index:
    if i not in subset_idx:
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
plt.savefig("Diagrams/robot_subset_map3.png", dpi = 150, bbox_inches = "tight")