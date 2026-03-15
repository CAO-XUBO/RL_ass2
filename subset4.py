import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Load in the robot location data
robots = pd.read_csv("processed_data/robot_locations_range.csv")

# High density subset
mixed_subset1 = robots.loc[(robots["longitude"] >= -120) & (robots["longitude"] <= -60) 
                         & (robots["latitude"] >= -90) & (robots["latitude"] <= -80)].copy()

mixed_subset2 = robots.loc[(robots["longitude"] >= 70) & (robots["longitude"] <= 130) 
                         & (robots["latitude"] >= -80) & (robots["latitude"] <= -70)].copy()

mixed_subset3 = robots.loc[(robots["longitude"] >= -120) & (robots["longitude"] <= -60) 
                         & (robots["latitude"] >= -80) & (robots["latitude"] <= -70)].copy()

# Output the subsets in CSV
mixed_subset1["subset"] = "mixed1"
mixed_subset2["subset"] = "mixed2"
mixed_subset3["subset"] = "mixed3"


subsets = pd.concat([mixed_subset1, mixed_subset2, mixed_subset3], ignore_index = True)
subsets.to_csv("processed_data/robot_subsets4.csv", index = False)

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
    x = mixed_subset1["longitude"], 
    y = mixed_subset1["latitude"],
    c = "red",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"Mixed density 1 (n={len(mixed_subset1)})"
)


# Plot median-density subset as yellow points
ax.scatter(
    x = mixed_subset2["longitude"], 
    y = mixed_subset2["latitude"],
    c = "gold",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"Mixed density 2 (n={len(mixed_subset2)})"
)

# Plot low-density subset as green points
ax.scatter(
    x = mixed_subset3["longitude"], 
    y = mixed_subset3["latitude"],
    c = "green",
    s = 8,
    alpha = 0.8,
    transform = ccrs.PlateCarree(),
    label = f"Mixed density 3 (n={len(mixed_subset3)})"
)

# Plot all other robots as gray points
all_subset_idx = mixed_subset1.index.union(mixed_subset2.index).union(mixed_subset3.index)
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
plt.savefig("Diagrams/robot_subset_map4.png", dpi = 150, bbox_inches = "tight")