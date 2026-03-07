import math
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from Hyperparameter import *
from cost_calculator import calculate_single_station_cost, calculate_global_expected_cost

def data_load(data_filepath):
    robot_data = pd.read_csv(data_filepath, index_col='index')
    return robot_data

def assign_best_available_station(robot_id, station_locations, station_current_counts, robot_data):
    """
    Identify the most optimal location for a given robot at the present moment.
    """
    rx, ry = robot_data.loc[robot_id, ['longitude', 'latitude']]
    robot_range = robot_data.loc[robot_id, 'range']

    # Calculate the distances to all sites and sort them
    candidates = []
    for s_id, (sx, sy) in station_locations.items():
        dist = math.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)
        if dist <= robot_range:
            candidates.append((dist, s_id))

    # Sort by distance in ascending order
    candidates.sort()

    # Locate the nearest station with available space in sequence
    for dist, s_id in candidates:
        if station_current_counts[s_id] < CAPACITY_LIMIT:
            return s_id, dist

    return -1, None


def run_greedy_construction(robot_data):
    """
    Greedy Website Construction Algorithm
    """

    # Initialization
    station_locations = {}
    station_current_counts = {}
    allocations = {r_id: -1 for r_id in robot_data.index}

    unassigned_robots = list(robot_data.index)
    station_counter = 0
    max_capacity = STATION_CHARGER_LIMIT * CHARGER_ROBOT_LIMIT

    print(f"Loaded {len(unassigned_robots)} robot data")

    while len(unassigned_robots) > 0:
        best_savings = 0
        best_candidate_xy = None
        best_candidate_robots = []

        unassigned_data = robot_data.loc[unassigned_robots]
        coords = unassigned_data[['longitude', 'latitude']].values
        ranges = unassigned_data['range'].values
        ids = unassigned_data.index.values

        for i, seed_id in enumerate(ids):
            seed_coord = coords[i].reshape(1, 2)

            # Distance from the seed to all other unassigned robots
            dist_to_seed = cdist(seed_coord, coords)[0]

            valid_mask = dist_to_seed <= ranges
            valid_indices = np.where(valid_mask)[0]

            sorted_valid_indices = valid_indices[np.argsort(dist_to_seed[valid_indices])]
            top_k_indices = sorted_valid_indices[:max_capacity]

            group_ids = ids[top_k_indices]
            group_coords = coords[top_k_indices]
            group_ranges = ranges[top_k_indices]

            # Center of Gravity
            center_x = np.mean(group_coords[:, 0])
            center_y = np.mean(group_coords[:, 1])
            center_coord = np.array([[center_x, center_y]])

            # Flight distance
            dist_to_center = cdist(group_coords, center_coord).flatten()

            # Calculate the cost
            station_cost = calculate_single_station_cost(dist_to_center, group_ranges)
            penalty_avoided = len(group_ids) * RESCUE_COST
            savings = penalty_avoided - station_cost

            # Record the maximum profit across the entire map
            if savings > best_savings:
                best_savings = savings
                best_candidate_xy = (center_x, center_y)

                valid_final_mask = dist_to_center <= group_ranges
                best_candidate_robots = group_ids[valid_final_mask][:max_capacity]

        if best_savings > 0 and len(best_candidate_robots) > 0:
            station_id = f"S_{station_counter}"
            station_locations[station_id] = best_candidate_xy

            for r_id in best_candidate_robots:
                allocations[r_id] = station_id
                unassigned_robots.remove(r_id)

            station_current_counts[station_id] = len(best_candidate_robots)
            print(f"Build {station_id}: location:(longitude:{best_candidate_xy[0]:.2f}, latitude:{best_candidate_xy[1]:.2f}), "
                  f"Capacity {len(best_candidate_robots)}/16, Savings £{best_savings:.2f}")
            station_counter += 1
        else:
            print("Greedy search termination.")
            break

    # Sweep Phase
    print(f"Sweep {len(unassigned_robots)} robot.")
    saved_in_sweep = 0
    for r_id in unassigned_robots[:]:
        s_id, dist = assign_best_available_station(r_id, station_locations, station_current_counts, robot_data)
        if s_id != -1:
            allocations[r_id] = s_id
            station_current_counts[s_id] += 1
            unassigned_robots.remove(r_id)
            saved_in_sweep += 1


    print(f"Total number of charging stations constructed: {len(station_locations)}")
    print(f"Successful Charging Robot: {len(robot_data) - len(unassigned_robots)}")
    print(f"The number of robot that must call for assistance: {len(unassigned_robots)}")

    return station_locations, allocations, station_current_counts


if __name__ == '__main__':

    data_path = "processed_data/robot_locations_range.csv"

    out_dir = "results"
    stations_out_path = f"{out_dir}/stations_1b.csv"
    allocations_out_path = f"{out_dir}/allocations_1b.csv"

    robot_data = data_load(data_path)

    final_stations, final_allocations, final_counts = run_greedy_construction(robot_data)
    final_total_cost = calculate_global_expected_cost(final_stations, final_allocations, robot_data)

    print(f"The Total Expected Daily Cost is: £{final_total_cost:,.2f}")
    print(f"Total Stations Built: {len(final_stations)}")

    stations_df = pd.DataFrame.from_dict(final_stations, orient='index', columns=['longitude', 'latitude'])
    stations_df.index.name = 'station_id'

    stations_df['robot_count'] = pd.Series(final_counts)
    stations_df.to_csv(stations_out_path)

    allocations_df = pd.DataFrame.from_dict(final_allocations, orient='index', columns=['station_id'])
    allocations_df.index.name = 'robot_id'
    allocations_df.to_csv(allocations_out_path)