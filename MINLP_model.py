import xpress as xp
import pandas as pd
import numpy as np
from Hyperparameter import *
subsets_df = pd.read_csv('processed_data/robot_subsets.csv')

densities = ['high', 'low', 'median']
results = {}

def solve_minlp_for_subset(df_subset, name):
    print(f"\n--- Starting Optimization for {name} density ---")
    # data preparation
    x_coords = df_subset['longitude'].values
    y_coords = df_subset['latitude'].values
    r_i = df_subset['range'].values
    n_robots = len(df_subset)
    I = range(n_robots)
    # Set the number of stations to 8 as per your requirement
    n_stations =  8
    J = range(n_stations)

    # Calculate the probability of rescue failure for each robot based on its range and the minimum required range
    p_i = np.exp(-(LAMBDA**2) * (r_i - R_MIN)**2)
    
    # Calculate the big M constant for the distance constraints
    X_min, X_max = x_coords.min(), x_coords.max()
    Y_min, Y_max = y_coords.min(), y_coords.max()
    D_max = np.sqrt((X_max - X_min)**2 + (Y_max - Y_min)**2)
    M = D_max - R_MIN

    prob = xp.problem(f"Antarctica_{name}")
    
    # Decision variables
    X_j = xp.vars(J, name='X_j', lb=X_min, ub=X_max)
    Y_j = xp.vars(J, name='Y_j', lb=Y_min, ub=Y_max)
    v_j = xp.vars(J, vartype=xp.integer, lb=0, ub=STATION_CHARGER_LIMIT, name='v_j')
    z_j = xp.vars(J, vartype=xp.binary, name='z_j')
    w_ij = xp.vars(I, J, vartype=xp.binary, name='w_ij')
    h_ij = xp.vars(I, J, vartype=xp.binary, name='h_ij')
    
    prob.addVariable(X_j, Y_j, v_j, z_j, w_ij, h_ij)

    # Objective function
    obj = xp.Sum(BUILD_COST * z_j[j] for j in J) + \
          xp.Sum(MAINTAIN_COST * v_j[j] for j in J) + \
          xp.Sum(p_i[i] * RESCUE_COST * h_ij[i,j] for i in I for j in J) + \
          xp.Sum(p_i[i] * CHARGE_COST * R_MAX * w_ij[i,j] for i in I for j in J)
    
    prob.setObjective(obj, sense=xp.minimize)

    # Constraints
    prob.addConstraint(xp.Sum(w_ij[i,j] for j in J) == 1 for i in I) 
    prob.addConstraint(xp.Sum(w_ij[i,j] for i in I) <= CHARGER_ROBOT_LIMIT * v_j[j] for j in J) 
    prob.addConstraint(v_j[j] <= STATION_CHARGER_LIMIT * z_j[j] for j in J) 

    for i in I:
        for j in J:
            dist = xp.sqrt((X_j[j] - x_coords[i])**2 + (Y_j[j] - y_coords[i])**2) 
            prob.addConstraint(dist - r_i[i] <= M * h_ij[i,j] + M * (1 - w_ij[i,j])) 
            prob.addConstraint(h_ij[i,j] <= w_ij[i,j]) 

    prob.controls.timelimit = 300 
    prob.solve()
    
    return prob, X_j, Y_j, z_j, v_j, x_coords, y_coords

# # Results
import os
all_station_data = []
summary_results = []  


for density in densities:
    df_sub = subsets_df[subsets_df['subset'] == density]
    if not df_sub.empty:
       
        prob, X_j, Y_j, z_j, v_j, rx, ry = solve_minlp_for_subset(df_sub, density)
        
        sol_status = prob.attributes.solstatus
        
        if sol_status in [xp.SolStatus.FEASIBLE, xp.SolStatus.OPTIMAL]:
            total_cost = prob.attributes.objval
            active_stations_count = 0
            total_chargers = 0
            
            for j in range(len(X_j)):
                z_val = prob.getSolution(z_j[j])
                if z_val > 0.5:
                    active_stations_count += 1
                    num_chargers = int(round(prob.getSolution(v_j[j])))
                    total_chargers += num_chargers
                    
                    all_station_data.append({
                        'Density_Type': density.capitalize(),
                        'Station_Index': j,
                        'X_Coord': prob.getSolution(X_j[j]),
                        'Y_Coord': prob.getSolution(Y_j[j]),
                        'Chargers_Built': num_chargers,
                        'Capacity_Robots': num_chargers * 2 
                    })
            
            avg_v = total_chargers / active_stations_count if active_stations_count > 0 else 0
            
            summary_results.append({
                'Scenario': density.capitalize(),
                'Cost': f"£{total_cost:,.2f}",
                'Stations': active_stations_count,
                'Avg_Chargers': f"{avg_v:.1f}"
            })
        else:
            summary_results.append({
                'Scenario': density.capitalize(),
                'Cost': f"Status: {sol_status}",
                'Stations': "-",
                'Avg_Chargers': "-"
            })
print("\n" + "="*70)
print(f"{'1(a) FINAL SUMMARY RESULTS':^70}")
print("="*70)
print(f"{'DENSITY SCENARIO':<18} | {'TOTAL COST':<15} | {'STATIONS':<10} | {'AVG CHARGERS':<12}")
print("-" * 70)
for res in summary_results:
    print(f"{res['Scenario']:<18} | {res['Cost']:>15} | {res['Stations']:<10} | {res['Avg_Chargers']:<12}")
print("="*70)
output_dir = "MINLP_results"
if all_station_data:
    results_df = pd.DataFrame(all_station_data)
    results_df = results_df.sort_values(by=['Density_Type', 'Station_Index'])
    csv_path = os.path.join(output_dir, "MINLP_subset_results_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nStation details saved to: {csv_path}")
print("="*75)