import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Hyperparameter import R_MIN

# 1. Load scenario data
scenarios_path = "origin_data/range_scenarios.csv"
scenarios_df = pd.read_csv(scenarios_path, index_col=0)

# Define sensitivity parameters (lambda) to test
test_lambdas = [0.005, 0.008, 0.012, 0.015, 0.020]
expected_charging_robots = []

print("Analyzing expected daily charging demand...")

# 2. Calculate the expected daily number of robots requiring a charge for each lambda
for l_val in test_lambdas:
    total_expected_demand = 0.0
    
    for s_idx in scenarios_df.columns:
        daily_ranges = scenarios_df[s_idx].values
        
        # Calculate the charging probability p_is for each robot
        p_is_array = np.exp(- (l_val**2) * ((daily_ranges - R_MIN)**2))
        
        # Sum the probabilities to get the expected daily demand
        daily_demand = p_is_array.sum()
        total_expected_demand += daily_demand
        
    # Compute the average expected demand across all scenarios
    avg_demand = total_expected_demand / len(scenarios_df.columns)
    expected_charging_robots.append(avg_demand)
    print(f"Lambda: {l_val:.3f} | Expected Charging Robots: {avg_demand:.0f} / 1072")

# 3. Generate a bar chart for expected demand
plt.figure(figsize=(9, 5))
bars = plt.bar([str(l) for l in test_lambdas], expected_charging_robots, color='skyblue', edgecolor='black')

# Annotate bars with expected robot counts and theoretical minimum station requirements
for bar, demand in zip(bars, expected_charging_robots):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f"{demand:.0f} robots\n(~{int(np.ceil(demand/16))} stations)", 
             ha='center', va='bottom', fontsize=10)

# Add a baseline for the total fleet size
plt.axhline(y=1072, color='red', linestyle='--', label='Total Robots (1072)')

plt.title(r'Expected Daily Charging Demand vs. $\lambda$', fontsize=14, fontweight='bold')
plt.xlabel(r'Weather Severity ($\lambda$)', fontsize=12)
plt.ylabel('Expected Number of Charging Robots', fontsize=12)
plt.ylim(0, 1200)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()

out_path = "results/demand_analysis_plot.png"
os.makedirs("results", exist_ok=True)
plt.savefig(out_path, dpi=300)
print(f"Plot saved to {out_path}")