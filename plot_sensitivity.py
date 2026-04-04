import matplotlib.pyplot as plt


# 1. Input Data
lambdas = [0.005, 0.008, 0.012, 0.015, 0.020]
baseline_costs = [645077.52, 639307.50, 631369.84, 626260.34, 619838.24]
optimized_costs = [644932.35, 617892.50, 510135.61, 420683.59, 300715.30]
saved_costs = [145.16, 21415.01, 121234.23, 205576.75, 319122.94]
final_stations = [67, 57, 40, 32, 22]


# 2. Plot 1: Cost Comparison Line Chart
plt.figure(figsize=(10, 5))

plt.plot(lambdas, baseline_costs, marker='o', linestyle='--', color='#e63946', linewidth=2, label='Deterministic Baseline')
plt.plot(lambdas, optimized_costs, marker='s', linestyle='-', color='#1d3557', linewidth=2, label='Stochastic Robust')

# Annotate cost savings
for i in range(len(lambdas)):
    plt.annotate(f"Saved\n£{saved_costs[i]:,.0f}", 
                 (lambdas[i], optimized_costs[i]), 
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center', fontsize=9, color='#2a9d8f', fontweight='bold')

plt.title(r'Expected Cost vs. Weather Severity ($\lambda$)', fontsize=13, fontweight='bold')
plt.xlabel(r'Weather Severity Parameter ($\lambda$)', fontsize=11, fontweight='bold')
plt.ylabel('Expected Total Cost (£)', fontsize=11, fontweight='bold')
plt.xticks(lambdas, [f"{l:.3f}" for l in lambdas]) 
plt.ylim(200000, 750000)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

cost_filename = 'plot_cost_sensitivity.png'
plt.savefig(cost_filename, dpi=300, bbox_inches='tight')
plt.close() 
print(f"Plot 1 successfully generated: {cost_filename}")


# 3. Plot 2: Infrastructure Sizing Bar Chart
plt.figure(figsize=(7, 5))

plt.bar(lambdas, final_stations, width=0.0015, color='#457b9d', edgecolor='black', alpha=0.8, label='Optimal Stations Kept')
plt.axhline(y=67, color='#e63946', linestyle='--', linewidth=2, label='Deterministic Baseline (67)')

# Annotate station counts
for i in range(len(lambdas)):
    plt.text(lambdas[i], final_stations[i] + 2, f"{final_stations[i]}", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Optimal Infrastructure Sizing vs. Weather', fontsize=13, fontweight='bold')
plt.xlabel(r'Weather Severity Parameter ($\lambda$) $\rightarrow$ Milder Weather', fontsize=11, fontweight='bold')
plt.ylabel('Number of Stations', fontsize=11, fontweight='bold')
plt.ylim(0, 85)
plt.xticks(lambdas, [f"{l:.3f}" for l in lambdas]) 
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()

stations_filename = 'plot_stations_sensitivity.png'
plt.savefig(stations_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot 2 successfully generated: {stations_filename}")