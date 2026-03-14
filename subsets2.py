import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Load in the robot location data
robots = pd.read_csv("processed_data/robot_locations_range.csv")

# High density subset
high_subset = robots.loc[(robots["longitude"] >= -90) & (robots["longitude"] <= -30) 
                         & (robots["latitude"] >= -70) & (robots["latitude"] <= -60)].copy()

mixed_subset = robots.loc[(robots["longitude"] >= 90) & (robots["longitude"] <= 150) 
                         & (robots["latitude"] >= -80) & (robots["latitude"] <= -70)].copy()

low_subset = robots.loc[(robots["longitude"] >= 0) & (robots["longitude"] <= 60) 
                         & (robots["latitude"] >= -80) & (robots["latitude"] <= -70)].copy()

# Output the subsets in CSV
high_subset["subset"] = "high"
low_subset["subset"] = "low"
mixed_subset["subset"] = "mixed"

subsets = pd.concat([high_subset, mixed_subset, low_subset], ignore_index = True)
subsets.to_csv("processed_data/robot_subsets2.csv", index = False)

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
    x = mixed_subset["longitude"], 
    y = mixed_subset["latitude"],
    c = "gold",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"Mixed density (n={len(mixed_subset)})"
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
all_subset_idx = high_subset.index.union(low_subset.index).union(mixed_subset.index)
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
plt.savefig("Diagrams/robot_subset_map2.png", dpi = 150, bbox_inches = "tight")