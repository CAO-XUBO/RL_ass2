import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from Hyperparameter import *

def data_load(data_filepath, target_subset=None, is_subset=False):
    """Load robot data from CSV."""
    robot_data = pd.read_csv(data_filepath, index_col='index')
    if is_subset:
        target_robot_data = robot_data[robot_data['subset'] == target_subset].copy()
    else:
        target_robot_data = robot_data.copy()
    return target_robot_data


def calculate_stochastic_cost(stations_dict, robot_data, scenarios_df, lambda_val):
    """
    Evaluate the second-stage expected cost based on stochastic range scenarios.
    """
    num_stations = len(stations_dict)
    if num_stations == 0:
        return float('inf')
        
    # First-stage fixed costs (construction and maintenance)
    fixed_cost = num_stations * BUILD_COST + num_stations * STATION_CHARGER_LIMIT * MAINTAIN_COST
    total_scenario_cost = 0.0
    num_scenarios = len(scenarios_df.columns)
    robot_coords = {r_id: (row['longitude'], row['latitude']) for r_id, row in robot_data.iterrows()}
    
    # Second-stage expected operational costs
    for s_idx in scenarios_df.columns:
        daily_cost = 0.0
        daily_ranges = scenarios_df[s_idx] 
        station_capacities = {s_id: CAPACITY_LIMIT for s_id in stations_dict.keys()}
        
        robots_today = []
        for r_id in robot_data.index:
            r_is = daily_ranges[r_id]
            # Calculate the probability of battery depletion based on lambda_val
            p_is = math.exp(- (lambda_val**2) * ((r_is - R_MIN)**2))
            robots_today.append({'id': r_id, 'range': r_is, 'prob': p_is})
            
        # Prioritize robots with the highest depletion probability
        robots_today.sort(key=lambda x: x['prob'], reverse=True)
        
        for rob in robots_today:
            r_id = rob['id']
            r_is = rob['range']
            p_is = rob['prob']
            rx, ry = robot_coords[r_id]
            
            best_station = None
            min_expected_cost = float('inf')
            
            for s_id, (sx, sy) in stations_dict.items():
                if station_capacities[s_id] <= 0:
                    continue 
                    
                dist = math.sqrt((rx - sx)**2 + (ry - sy)**2)
                
                if dist <= r_is:
                    cost = p_is * CHARGE_COST * ((R_MAX - r_is) + dist)
                else:
                    cost = p_is * (RESCUE_COST + CHARGE_COST * (R_MAX - r_is))
                    
                if cost < min_expected_cost:
                    min_expected_cost = cost
                    best_station = s_id
                    
            if best_station is not None:
                station_capacities[best_station] -= 1
                daily_cost += min_expected_cost
            else:
                # Capacity limit reached; incur rescue penalty
                daily_cost += p_is * (RESCUE_COST + CHARGE_COST * (R_MAX - r_is))
                
        total_scenario_cost += daily_cost
        
    expected_total_cost = fixed_cost + (total_scenario_cost / num_scenarios)
    return expected_total_cost


def run_stochastic_local_search(robot_data, scenarios_df, initial_stations_dict, lambda_val):
    """
    Execute heuristic local search with demand-driven dynamic capacity bounds.
    """
    stations_dict = initial_stations_dict.copy()
    current_cost = calculate_stochastic_cost(stations_dict, robot_data, scenarios_df, lambda_val)
    
    # Calculate demand-driven dynamic capacity lower bound
    total_expected_demand = 0.0
    for s_idx in scenarios_df.columns:
        daily_ranges = scenarios_df[s_idx].values
        p_is_array = np.exp(- (lambda_val**2) * ((daily_ranges - R_MIN)**2))
        total_expected_demand += p_is_array.sum()
    avg_demand = total_expected_demand / len(scenarios_df.columns)
    
    # Theoretical minimum stations required + 10% safety buffer
    base_needed = math.ceil(avg_demand / CAPACITY_LIMIT)
    dynamic_min_stations = math.ceil(base_needed * 1.10) 
    
    print(f"  [Info] Average expected demand: {avg_demand:.0f} robots")
    print(f"  [Info] Dynamic station lower bound set to: {dynamic_min_stations} (includes 10% buffer)")

    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1
        print(f"  [Iter {iteration}] Searching for improvements...")

        # Operator 1: Relocate stations to cluster centroids
        for s_id in list(stations_dict.keys()):
            sx, sy = stations_dict[s_id]
            distances = []
            for r_id in robot_data.index:
                rx, ry = robot_data.loc[r_id, ['longitude', 'latitude']]
                distances.append((math.sqrt((rx-sx)**2 + (ry-sy)**2), rx, ry))
            
            distances.sort(key=lambda x: x[0])
            nearest_k = distances[:CAPACITY_LIMIT]
            
            new_x = np.mean([d[1] for d in nearest_k])
            new_y = np.mean([d[2] for d in nearest_k])
            
            old_xy = stations_dict[s_id]
            stations_dict[s_id] = (new_x, new_y)
            
            new_cost = calculate_stochastic_cost(stations_dict, robot_data, scenarios_df, lambda_val)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
                print(f"      Relocate: Station {s_id} shifted. New expected cost: GBP {current_cost:,.2f}")
            else:
                stations_dict[s_id] = old_xy 

        # Operator 2: Drop redundant stations (protected by dynamic lower bound)
        if not improved:
            for s_id in list(stations_dict.keys()):
                # Enforce dynamic capacity lower bound
                if len(stations_dict) <= dynamic_min_stations:
                    print(f"      [Drop Blocked] Reached dynamic lower bound of {dynamic_min_stations} stations.")
                    break 
                
                old_xy = stations_dict.pop(s_id)
                new_cost = calculate_stochastic_cost(stations_dict, robot_data, scenarios_df, lambda_val)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    print(f"      Drop: Station {s_id} permanently closed. New expected cost: GBP {current_cost:,.2f}")
                    break
                else:
                    stations_dict[s_id] = old_xy  # Cost increased; revert the drop
                    
    print(f"  Final Stations Count: {len(stations_dict)}")
    return stations_dict, current_cost


if __name__ == '__main__':
    # Configuration and file paths
    robot_data_path = "processed_data/robot_locations_range.csv" 
    scenarios_path = "origin_data/range_scenarios.csv"
    stations_in_path = "results/local_search_deterministic/local_search_deterministic_full/stations_local_search.csv"
    out_dir = "results/stochastic_sensitivity"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Loading datasets...")
    robot_data = data_load(robot_data_path, is_subset=False)
    scenarios_df = pd.read_csv(scenarios_path, index_col=0)
    
    stations_df = pd.read_csv(stations_in_path, index_col='station_id')
    initial_stations_dict = {str(idx): (row['longitude'], row['latitude']) for idx, row in stations_df.iterrows()}
    
    # Sensitivity Analysis Loop
    test_lambdas = [0.004, 0.008, 0.012, 0.016, 0.020]
    
    baseline_costs = []
    optimized_costs = []
    saved_moneys = []
    final_station_counts = []
    
    print("\nStarting Comprehensive Sensitivity Analysis...")
    for l_val in test_lambdas:
        print(f"\n{'='*50}")
        print(f"Evaluating Lambda = {l_val}")
        print(f"{'='*50}")
        
        # 1. Evaluate deterministic baseline under stochastic conditions
        base_cost = calculate_stochastic_cost(initial_stations_dict, robot_data, scenarios_df, l_val)
        baseline_costs.append(base_cost)
        print(f"Deterministic Baseline (Model I) Cost: GBP {base_cost:,.2f} (67 stations)")
        
        # 2. Run stochastic robust optimization
        final_stations, opt_cost = run_stochastic_local_search(robot_data, scenarios_df, initial_stations_dict, l_val)
        optimized_costs.append(opt_cost)
        final_station_counts.append(len(final_stations))
        
        # 3. Record cost savings
        saved = base_cost - opt_cost
        saved_moneys.append(saved)
        print(f"\nTotal Cost Saved by Stochastic Model (Model II): GBP {saved:,.2f}")

        # 4. Save final station locations and typical allocations for this lambda
        stations_out_df = pd.DataFrame.from_dict(final_stations, orient='index', columns=['longitude', 'latitude'])
        stations_out_df.index.name = 'station_id'
        stations_out_csv = f"{out_dir}/stations_lambda_{l_val}.csv"
        stations_out_df.to_csv(stations_out_csv)
        print(f"  --> Saved Stations to {stations_out_csv}")

        # Generate typical allocations for this lambda
        avg_ranges = scenarios_df.mean(axis=1) 
        robot_coords = {r_id: (row['longitude'], row['latitude']) for r_id, row in robot_data.iterrows()}
        station_capacities = {s_id: CAPACITY_LIMIT for s_id in final_stations.keys()}
        
        robots_typical = []
        for r_id in robot_data.index:
            r_avg = avg_ranges[r_id]
            p_fail = math.exp(- (l_val**2) * ((r_avg - R_MIN)**2))
            robots_typical.append({'id': r_id, 'range': r_avg, 'prob': p_fail})
            
        robots_typical.sort(key=lambda x: x['prob'], reverse=True)
        
        typical_allocations = {}
        for rob in robots_typical:
            r_id = rob['id']
            r_is = rob['range']
            p_is = rob['prob']
            rx, ry = robot_coords[r_id]
            
            best_station = None
            min_expected_cost = float('inf')
            
            for s_id, (sx, sy) in final_stations.items():
                if station_capacities[s_id] <= 0:
                    continue 
                dist = math.sqrt((rx - sx)**2 + (ry - sy)**2)
                if dist <= r_is:
                    cost = p_is * CHARGE_COST * ((R_MAX - r_is) + dist)
                else:
                    cost = p_is * (RESCUE_COST + CHARGE_COST * (R_MAX - r_is))
                    
                if cost < min_expected_cost:
                    min_expected_cost = cost
                    best_station = s_id
                    
            if best_station is not None:
                station_capacities[best_station] -= 1
                typical_allocations[r_id] = best_station
            else:
                typical_allocations[r_id] = "Rescue_Needed"
                
        allocations_out_df = pd.DataFrame.from_dict(typical_allocations, orient='index', columns=['station_id'])
        allocations_out_df.index.name = 'robot_id'
        alloc_out_csv = f"{out_dir}/allocations_lambda_{l_val}_typical.csv"
        allocations_out_df.to_csv(alloc_out_csv)
        print(f"  --> Saved Typical Allocations to {alloc_out_csv}")

    # Save results and generate plots
    results_df = pd.DataFrame({
        'Lambda': test_lambdas,
        'Baseline_Cost': baseline_costs,
        'Optimized_Cost': optimized_costs,
        'Cost_Saved': saved_moneys,
        'Final_Stations': final_station_counts
    })
    results_df.to_csv(f"{out_dir}/ultimate_sensitivity_results.csv", index=False)
    print(f"\nResults successfully saved to {out_dir}/ultimate_sensitivity_results.csv")

    # Plot generation
    plt.figure(figsize=(10, 6))
    
    plt.plot(test_lambdas, baseline_costs, marker='o', linestyle='--', color='#e63946', label='Deterministic Baseline (67 stations)')
    plt.plot(test_lambdas, optimized_costs, marker='s', linestyle='-', color='#1d3557', label='Stochastic Robust (Dynamic stations)')
    
    plt.title(r'Sensitivity Analysis: Expected Cost vs. Weather Severity ($\lambda$)', fontsize=14, fontweight='bold')
    plt.xlabel(r'Weather Severity Parameter ($\lambda$)', fontsize=12)
    plt.ylabel('Expected Total Cost (GBP)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Annotate cost savings and final station counts
    for i in range(len(test_lambdas)):
        annot_text = f"Saved: £{saved_moneys[i]:,.0f}\n({final_station_counts[i]} stations)"
        plt.annotate(annot_text, 
                     (test_lambdas[i], optimized_costs[i]), 
                     textcoords="offset points", 
                     xytext=(0, -35), 
                     ha='center', fontsize=9, color='#2a9d8f')

    plt.tight_layout()
    plot_path = f"{out_dir}/ultimate_lambda_sensitivity_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Sensitivity plot saved to {plot_path}")