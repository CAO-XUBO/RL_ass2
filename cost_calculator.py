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