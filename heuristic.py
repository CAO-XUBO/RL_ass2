import math
from Hyperparameter import *

def calculate_single_station_cost(distances, max_ranges):
    '''
    Calculate the true total cost of constructing a single charging station
    '''
    valid_distances = []
    penalty_count = 0  # Record the number of robots requiring fines to be paid

    # Range Check
    for d, r in zip(distances, max_ranges):
        if d <= r:
            valid_distances.append(d)
        else:
            penalty_count += 1

    # Capacity Check
    max_capacity = STATION_CHARGER_LIMIT * CHARGER_ROBOT_LIMIT

    if len(valid_distances) > max_capacity:
        valid_distances.sort()

        excess_count = len(valid_distances) - max_capacity
        penalty_count += excess_count

        valid_distances = valid_distances[:max_capacity]

    # Cost Accounting
    N_served = len(valid_distances)

    if N_served == 0:
        return BUILD_COST + (penalty_count * RESCUE_COST)

    num_chargers = math.ceil(N_served / CHARGER_ROBOT_LIMIT)

    # Total hardware costs
    build_and_maintain_cost = BUILD_COST + num_chargers * MAINTAIN_COST

    # Flight charging fee
    flight_cost = sum(valid_distances) * CHARGE_COST

    # Rescue penalty fee
    penalty_cost = penalty_count * RESCUE_COST

    # Aggregate total cost
    total_single_station_cost = build_and_maintain_cost + flight_cost + penalty_cost

    return total_single_station_cost


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