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

    # Expected dynamic costs
    expected_variable_cost = 0.0

    for d, r_i in zip(distances, max_ranges):
        # The probability that the robot requires charging today
        p_i = math.exp(- (LAMBDA ** 2) * ((r_i - R_MIN) ** 2))

        # Expected charging fee
        charge_cost_per_time = CHARGE_COST * R_MAX
        expected_variable_cost += p_i * charge_cost_per_time

        # Expected trailer fee
        if d > r_i:
            expected_variable_cost += p_i * RESCUE_COST

    # Total expect cost
    total_expected_cost = fixed_hardware_cost + expected_variable_cost

    return total_expected_cost

def calculate_global_expected_cost(station_locations, allocations, robot_data):
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
