import math
import numpy as np
from Hyperparameter import *
from scipy.spatial.distance import cdist

def calculate_single_station_cost(distances, max_ranges):
    '''
    Calculate the true total cost of constructing a single charging station
    '''

    N_served = len(distances)
    if N_served > CAPACITY_LIMIT:
        return float('inf')

    if N_served == 0:
        return 0.0

    # Hardware fixed costs
    num_chargers = math.ceil(N_served / CHARGER_ROBOT_LIMIT)
    fixed_hardware_cost = BUILD_COST + num_chargers * MAINTAIN_COST

    # Dynamic costs
    variable_cost = 0.0

    for d, r_i in zip(distances, max_ranges):
        if d <= r_i:
            charge_amount = R_MAX - r_i +d
            variable_cost += CHARGE_COST * charge_amount
        else:
            charge_amount = R_MAX - r_i
            variable_cost += CHARGE_COST * charge_amount
            variable_cost += RESCUE_COST

    # Total expect cost
    total_deterministic_cost = fixed_hardware_cost + variable_cost

    return total_deterministic_cost

def calculate_global_deterministic_cost(station_locations, allocations, robot_data):
    """
    Final calculate total cost for the entire settlement area
    """
    total_system_cost = 0.0

    # Group the robots assigned to base stations by base station
    station_groups = {s_id: [] for s_id in station_locations.keys()}
    for r_id, s_id in allocations.items():
        if s_id != -1 and s_id in station_groups:
            station_groups[s_id].append(r_id)

    for s_id, assigned_robots in station_groups.items():
        if len(assigned_robots) == 0:
            continue

        sx, sy = station_locations[s_id]
        station_coord = np.array([[sx, sy]])

        group_data = robot_data.loc[assigned_robots]
        group_coords = group_data[['longitude', 'latitude']].values
        group_ranges = group_data['range'].values

        dist_to_station = cdist(group_coords, station_coord).flatten()

        station_cost = calculate_single_station_cost(dist_to_station, group_ranges)
        total_system_cost += station_cost

    return total_system_cost

def evaluate_performance(robot_data, final_stations, final_total_cost, execution_time, target_subset, is_subset):
    """
    Evaluating the performance of heuristic algorithms
    """
    total_robots = len(robot_data)
    total_built_stations = len(final_stations)

    optimal_baselines = {
        "high": 26396.04,
        "low": 26512.19,
        "median": 26460.37,
    }

    print("\n" + "=" * 60)
    print(f"{'HEURISTIC PERFORMANCE EVALUATION':^60}")
    print("=" * 60)
    print(f"Dataset              : {'Subset - ' + target_subset.capitalize() if is_subset else 'Full Region (1072)'}")
    print(f"Total Robots Covered : {total_robots}")
    print(f"Total Stations Built : {total_built_stations}")
    print("-" * 60)

    f_x_h = final_total_cost
    print(f"Heuristic Cost f(X^H): £{f_x_h:,.2f}")

    if is_subset and target_subset.lower() in optimal_baselines:
        f_x_star = optimal_baselines[target_subset.lower()]
        rpd = ((f_x_h - f_x_star) / f_x_star) * 100
        print(f"Optimal Cost   f(X*): £{f_x_star:,.2f}")
        print(f"RPD (Gap)            : {rpd:.2f}%")
    else:
        print("Optimal Cost   f(X*): Unknown")
        print("RPD (Gap)            : N/A")

    print(f"Execution Time       : {execution_time:.4f} seconds")
    print("=" * 60 + "\n")
