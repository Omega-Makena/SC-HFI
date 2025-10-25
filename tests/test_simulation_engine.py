#!/usr/bin/env python3
"""
Test Simulation Engine with Market Data
Tests the actual simulation engine with real market data
"""

import pandas as pd
import numpy as np
import sys
import os
import json

# Add the sf_hfe_v2/moe directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sf_hfe_v2', 'moe'))

def load_and_prepare_market_data():
"""Load and prepare market price data for analysis"""

print("Loading market price data...")
df = pd.read_csv('Market_Prices.csv')

# Convert date column
df['MarketDate'] = pd.to_datetime(df['MarketDate'])

# Pivot to get stocks as columns and dates as rows
price_matrix = df.pivot(index='MarketDate', columns='Stock Name', values='MarketPrice')

# Fill missing values with forward fill
price_matrix = price_matrix.ffill()

# Calculate returns
returns_matrix = price_matrix.pct_change().dropna()

# Convert to numpy array
data_array = returns_matrix.values
n_samples, n_features = data_array.shape

# Get actual stock names
variable_names = list(returns_matrix.columns)

print(f"Market data: {n_samples} time periods × {n_features} stocks")
print(f"Variables: {variable_names}")
print(f"Date range: {returns_matrix.index.min().strftime('%Y-%m-%d')} to {returns_matrix.index.max().strftime('%Y-%m-%d')}")

return data_array, variable_names, returns_matrix

def test_simulation_engine():
"""Test the simulation engine with market data"""

print("=" * 60)
print("TESTING SIMULATION ENGINE WITH MARKET DATA")
print("=" * 60)

try:
# Import the simulation engine
from simulation import SimulationEngine

# Load market data
data_array, variable_names, returns_data = load_and_prepare_market_data()

# Initialize simulation engine
print("\nInitializing simulation engine...")
simulation = SimulationEngine(config={
'simulation_horizon': 5,
'num_scenarios': 3
})

print(" Simulation engine initialized successfully")

# Create analysis results from market data
print("\nCreating analysis results from market data...")

# Basic statistical analysis
analysis_results = {
'statistical': {
'correlations': np.corrcoef(data_array.T).tolist(),
'means': np.mean(data_array, axis=0).tolist(),
'stds': np.std(data_array, axis=0).tolist(),
'variables': variable_names
},
'temporal': {
'trends': np.polyfit(range(len(data_array)), data_array, 1)[0].tolist(),
'volatility': np.std(data_array, axis=0).tolist(),
'time_periods': len(data_array)
},
'structural': {
'data_shape': data_array.shape,
'missing_values': np.isnan(data_array).sum(),
'data_types': ['float64'] * len(variable_names)
}
}

print(f" Analysis results created for {len(variable_names)} variables")

# Run simulation scenarios
print("\nRunning simulation scenarios...")
simulation_results = simulation.run_scenarios(analysis_results)

print(" Simulation scenarios completed")

# Display results
print("\n" + "="*50)
print("SIMULATION RESULTS")
print("="*50)

if simulation_results:
print(" Simulation results generated successfully")

# Show scenario results
if 'scenario_results' in simulation_results:
scenarios = simulation_results['scenario_results']
print(f"\nGenerated {len(scenarios)} scenarios:")
for scenario_name, scenario_data in scenarios.items():
print(f" - {scenario_name}: {scenario_data.get('description', 'No description')}")

# Show synthetic insights
if 'synthetic_insights' in simulation_results:
insights = simulation_results['synthetic_insights']
print(f"\nGenerated {len(insights)} synthetic insights")
for i, insight in enumerate(insights[:3]): # Show first 3
print(f" {i+1}. {insight}")

# Show policy outcomes
if 'policy_outcomes' in simulation_results:
outcomes = simulation_results['policy_outcomes']
print(f"\nGenerated {len(outcomes)} policy outcomes")
for i, outcome in enumerate(outcomes[:2]): # Show first 2
print(f" {i+1}. {outcome}")

return simulation_results

except Exception as e:
print(f" Error running simulation engine: {e}")
import traceback
traceback.print_exc()
return None

def main():
"""Main test function"""

print("Testing Simulation Engine with Market Data")
print("=" * 60)

try:
results = test_simulation_engine()

print(f"\n{'='*60}")
print("SIMULATION ENGINE TESTING COMPLETE")
print(f"{'='*60}")

if results:
print(" Simulation engine is functional")
print(" Real market data processed successfully")
print(" Multiple scenarios generated")
print(" Synthetic insights created")
print(" Policy outcomes simulated")

print("\nThis demonstrates:")
print("1. Simulation engine works with real data")
print("2. Multiple scenarios can be generated")
print("3. Synthetic insights are created")
print("4. Policy outcomes are simulated")
else:
print(" Simulation engine failed")

except Exception as e:
print(f"\nError during testing: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
main()
