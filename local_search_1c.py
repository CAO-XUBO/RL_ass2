import math
import numpy as np
import pandas as pd
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
    Local Search Heuristic using Alternating Location-Allocation.
    Operators: Relocate (continuous) and Penalty-Guided Swap (discrete).
    """
    current_cost = calculate_global_deterministic_cost(stations_dict, allocations_dict, robot_data)
    print(f"Initial Cost: GBP {current_cost:,.2f}")

    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1

        # Operator 1: Relocate stations to cluster centroid
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
                print(f"  [Iter {iteration}] Relocate: Station {s_id} shifted. New cost: GBP {current_cost:,.2f}")
            else:
                stations_dict[s_id] = old_xy

        # Operator 2: Penalty-guided best-improvement swap
        if not improved:
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
                    print(f"  [Iter {iteration}] Swap: Robots {r1} & {r2} exchanged. New cost: GBP {current_cost:,.2f}")

    print(f"Local search converged after {iteration} iterations.")
    print(f"Final Objective Cost: GBP {current_cost:,.2f}")
    
    return stations_dict, allocations_dict


if __name__ == '__main__':
    scenarios = [
        {"is_subset": True,  "target_subset": "low"},
        {"is_subset": True,  "target_subset": "median"},
        {"is_subset": True,  "target_subset": "high"},
        {"is_subset": False, "target_subset": "full"}
    ]

    for scenario in scenarios:
        is_subset = scenario["is_subset"]
        target_subset = scenario["target_subset"]

        scenario_name = f"SUBSET: {target_subset.upper()}" if is_subset else "FULL DATASET"
        print(f"\n--- Processing Scenario: {scenario_name} ---")

        if is_subset:
            data_path = "processed_data/robot_subsets.csv"
            out_dir = f"results/heuristic_deterministic_{target_subset}"
        else:
            data_path = "processed_data/robot_locations_range.csv" 
            out_dir = "results/heuristic_deterministic_full"

        stations_in_path = f"{out_dir}/stations_deterministic.csv"
        allocations_in_path = f"{out_dir}/allocations_deterministic.csv"

        print(f"Loading initial solution from {out_dir}...")
        try:
            robot_data = data_load(data_path, target_subset, is_subset)
            stations_df = pd.read_csv(stations_in_path, index_col='station_id')
            allocations_df = pd.read_csv(allocations_in_path, index_col='robot_id')
        except FileNotFoundError:
            print(f"Files not found in {out_dir}. Skipping scenario.")
            continue

        stations_dict = {str(idx): (row['longitude'], row['latitude']) for idx, row in stations_df.iterrows()}
        allocations_dict = {idx: str(row['station_id']) for idx, row in allocations_df.iterrows()}

        final_stations, final_allocations = run_local_search(robot_data, stations_dict, allocations_dict)

        stations_out_path = f"{out_dir}/stations_local_search.csv"
        allocations_out_path = f"{out_dir}/allocations_local_search.csv"

        stations_out_df = pd.DataFrame.from_dict(final_stations, orient='index', columns=['longitude', 'latitude'])
        stations_out_df.index.name = 'station_id'
        
        final_counts = {s: 0 for s in final_stations.keys()}
        for r, s in final_allocations.items():
            if s in final_counts:
                final_counts[s] += 1
                
        stations_out_df['robot_count'] = pd.Series(final_counts)
        stations_out_df.to_csv(stations_out_path)

        allocations_out_df = pd.DataFrame.from_dict(final_allocations, orient='index', columns=['station_id'])
        allocations_out_df.index.name = 'robot_id'
        allocations_out_df.to_csv(allocations_out_path)
        
        print(f"Saved optimized results to {out_dir}")