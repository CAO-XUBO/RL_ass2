import math
import os
import numpy as np
import pandas as pd
import copy
from Hyperparameter import *
from cost_calculator import calculate_global_deterministic_cost

def data_load(data_filepath, target_subset=None, is_subset=False):
    """Load robot data from CSV."""
    robot_data = pd.read_csv(data_filepath, index_col='index')
    if is_subset:
        target_robot_data = robot_data[robot_data['subset'] == target_subset].copy()
    else:
        target_robot_data = robot_data.copy()
    return target_robot_data

def run_local_search(robot_data, stations_dict, allocations_dict):
    """
    Streamlined Local Search Heuristic for Zero-Spare-Capacity Scenario.
    Operators: Relocate (Op 1) and Penalty-Guided Swap (Op 4).
    """
    print("\n--- Starting Streamlined Local Search ---")

    current_cost = calculate_global_deterministic_cost(stations_dict, allocations_dict, robot_data)
    print(f"Initial Cost: £{current_cost:,.2f}")

    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1
        print(f"\nIteration {iteration}...")

        # ==========================================
        # Operator 1: Relocate stations to the center of gravity
        # This is CRUCIAL to adjust coordinates after a successful Swap.
        # ==========================================
        for s_id in list(stations_dict.keys()):
            assigned_robots = [r for r, s in allocations_dict.items() if s == s_id]
            if not assigned_robots:
                continue
            
            group_coords = robot_data.loc[assigned_robots, ['longitude', 'latitude']].values
            new_x = np.mean(group_coords[:, 0])
            new_y = np.mean(group_coords[:, 1])
            
            old_xy = stations_dict[s_id]
            stations_dict[s_id] = (new_x, new_y)
            
            new_cost = calculate_global_deterministic_cost(stations_dict, allocations_dict, robot_data)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
                print(f"  [Relocate] Station {s_id} moved to new centroid. New cost: £{current_cost:,.2f}")
            else:
                stations_dict[s_id] = old_xy

        # ==========================================
        # Operator 4: Penalty-guided best-improvement swap
        # Breaks the capacity deadlock by swapping high-penalty robots.
        # ==========================================
        if not improved:
            print("  [Swap] Exploring penalty-guided 1-to-1 swaps...")
            
            pain_robots = []
            for r_id in robot_data.index:
                s_id = allocations_dict[r_id]
                sx, sy = stations_dict[s_id]
                rx, ry = robot_data.loc[r_id, ['longitude', 'latitude']]
                dist = math.sqrt((rx - sx)**2 + (ry - sy)**2)
                if dist > robot_data.loc[r_id, 'range']:
                    pain_robots.append(r_id)
            
            if pain_robots:
                best_swap_pair = None
                best_swap_cost = current_cost
                robot_ids = list(robot_data.index)
                
                for r1 in pain_robots:
                    s1 = allocations_dict[r1]
                    for r2 in robot_ids:
                        s2 = allocations_dict[r2]
                        if s1 == s2:
                            continue
                        
                        allocations_dict[r1] = s2
                        allocations_dict[r2] = s1
                        
                        new_cost = calculate_global_deterministic_cost(stations_dict, allocations_dict, robot_data)
                        
                        if new_cost < best_swap_cost:
                            best_swap_cost = new_cost
                            best_swap_pair = (r1, r2)
                        
                        allocations_dict[r1] = s1
                        allocations_dict[r2] = s2

                if best_swap_pair is not None:
                    r1, r2 = best_swap_pair
                    s1_best = allocations_dict[r1]
                    s2_best = allocations_dict[r2]
                    
                    allocations_dict[r1] = s2_best
                    allocations_dict[r2] = s1_best
                    
                    current_cost = best_swap_cost
                    improved = True
                    print(f"  [Swap] Swapped robots {r1} and {r2}. New cost: £{current_cost:,.2f}")

    print(f"\nLocal search converged after {iteration} iterations.")
    print(f"Final Objective Cost: £{current_cost:,.2f}")
    print("--- End of Local Search ---\n")
    
    return stations_dict, allocations_dict


if __name__ == '__main__':

    data_path = "processed_data/robot_locations_range.csv"
    
    print("Loading 1(b) initial solution...")
    robot_data = data_load(data_path, is_subset=False)
    stations_1b_df = pd.read_csv("results/heuristic_deterministic/heuristic_deterministic_full/stations_deterministic.csv", index_col='station_id')
    allocations_1b_df = pd.read_csv("results/heuristic_deterministic/heuristic_deterministic_full/allocations_deterministic.csv", index_col='robot_id')

    stations_dict = {str(idx): (row['longitude'], row['latitude']) for idx, row in stations_1b_df.iterrows()}
    allocations_dict = {idx: str(row['station_id']) for idx, row in allocations_1b_df.iterrows()}

    final_stations, final_allocations = run_local_search(robot_data, stations_dict, allocations_dict)

    out_dir = "results/local_search_deterministic"

    stations_out_path = f"{out_dir}/stations_local_search.csv"
    allocations_out_path = f"{out_dir}/allocations_local_search.csv"

    stations_df = pd.DataFrame.from_dict(final_stations, orient='index', columns=['longitude', 'latitude'])
    stations_df.index.name = 'station_id'
    
    final_counts = {s: 0 for s in final_stations.keys()}
    for r, s in final_allocations.items():
        if s in final_counts:
            final_counts[s] += 1
            
    stations_df['robot_count'] = pd.Series(final_counts)
    stations_df.to_csv(stations_out_path)

    allocations_df = pd.DataFrame.from_dict(final_allocations, orient='index', columns=['station_id'])
    allocations_df.index.name = 'robot_id'
    allocations_df.to_csv(allocations_out_path)
    
    print(f"Results successfully saved to {stations_out_path} and {allocations_out_path}")