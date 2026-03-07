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