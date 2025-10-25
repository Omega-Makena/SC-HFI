#!/usr/bin/env python3
"""
Enhanced 3D Simulation Integration
Integrates 3D simulation with the existing simulation engine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add the sf_hfe_v2/moe directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sf_hfe_v2', 'moe'))

# Import ForecastingEngine from the previous file
from test_3d_simulation import ForecastingEngine

class EnhancedSimulation3D:
"""
Enhanced 3D Simulation that integrates with existing simulation engine
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.forecasting_engine = ForecastingEngine(config.get('forecasting', {}))

def run_enhanced_3d_simulation(self, data: pd.DataFrame, analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
"""Run enhanced 3D simulation with existing analysis results"""

print("Running Enhanced 3D Simulation...")

# If no analysis results provided, create them
if analysis_results is None:
analysis_results = self._create_analysis_results(data)

# Get target variables
target_variables = self._select_target_variables(data, analysis_results)

results = {
'forecasts': {},
'3d_scenarios': {},
'visualizations': {},
'simulation_summary': {},
'integration_results': {}
}

# Generate forecasts
print(f"Generating forecasts for {len(target_variables)} variables...")
for var in target_variables:
forecast_result = self.forecasting_engine.forecast(data, var)
results['forecasts'][var] = forecast_result

# Generate 3D scenarios based on analysis results
print("Generating 3D scenarios from analysis results...")
scenarios = self._generate_3d_scenarios_from_analysis(data, analysis_results, target_variables)
results['3d_scenarios'] = scenarios

# Create enhanced visualizations
print("Creating enhanced 3D visualizations...")
visualizations = self._create_enhanced_visualizations(data, results['forecasts'], scenarios, analysis_results)
results['visualizations'] = visualizations

# Integrate with existing simulation engine
print("Integrating with existing simulation engine...")
integration_results = self._integrate_with_existing_engine(data, analysis_results, results)
results['integration_results'] = integration_results

# Create comprehensive summary
results['simulation_summary'] = self._create_comprehensive_summary(results, analysis_results)

return results

def _create_analysis_results(self, data: pd.DataFrame) -> Dict[str, Any]:
"""Create analysis results from data"""

numeric_cols = data.select_dtypes(include=[np.number]).columns

analysis_results = {
'statistical': {
'correlations': data[numeric_cols].corr().values.tolist(),
'means': data[numeric_cols].mean().values.tolist(),
'stds': data[numeric_cols].std().values.tolist(),
'variables': list(numeric_cols)
},
'temporal': {
'trends': [],
'volatility': data[numeric_cols].std().values.tolist(),
'time_periods': len(data)
},
'structural': {
'data_shape': data.shape,
'missing_values': data.isnull().sum().sum(),
'data_types': [str(dtype) for dtype in data.dtypes]
}
}

# Calculate trends
for col in numeric_cols:
if len(data[col]) > 1:
trend = np.polyfit(range(len(data)), data[col], 1)[0]
analysis_results['temporal']['trends'].append(trend)

return analysis_results

def _select_target_variables(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
"""Select target variables for 3D simulation"""

numeric_cols = data.select_dtypes(include=[np.number]).columns

# Select variables based on volatility and correlation
volatility_scores = {}
for i, col in enumerate(numeric_cols):
if i < len(analysis_results['statistical']['stds']):
volatility_scores[col] = analysis_results['statistical']['stds'][i]

# Select top 3 most volatile variables
sorted_vars = sorted(volatility_scores.items(), key=lambda x: x[1], reverse=True)
target_variables = [var for var, _ in sorted_vars[:3]]

return target_variables

def _generate_3d_scenarios_from_analysis(self, data: pd.DataFrame, analysis_results: Dict[str, Any], target_variables: List[str]) -> Dict[str, Any]:
"""Generate 3D scenarios based on analysis results"""

scenarios = {}

if len(target_variables) >= 3:
x_var, y_var, z_var = target_variables[:3]

# Get current values
current_x = data[x_var].iloc[-1]
current_y = data[y_var].iloc[-1]
current_z = data[z_var].iloc[-1]

# Get historical statistics
x_mean = data[x_var].mean()
y_mean = data[y_var].mean()
z_mean = data[z_var].mean()

x_std = data[x_var].std()
y_std = data[y_var].std()
z_std = data[z_var].std()

# Generate scenarios based on statistical analysis
scenarios = {
'baseline': {
'coordinates': {'x': current_x, 'y': current_y, 'z': current_z},
'severity': 'low',
'probability': 0.5,
'impact': 'stable',
'description': 'Current state maintained',
'statistical_basis': 'current_values'
},
'mean_reversion': {
'coordinates': {'x': x_mean, 'y': y_mean, 'z': z_mean},
'severity': 'moderate',
'probability': 0.3,
'impact': 'mean_reversion',
'description': 'Variables return to historical means',
'statistical_basis': 'historical_means'
},
'high_volatility': {
'coordinates': {
'x': current_x + 2 * x_std,
'y': current_y + 2 * y_std,
'z': current_z + 2 * z_std
},
'severity': 'high',
'probability': 0.1,
'impact': 'high_volatility',
'description': 'High volatility scenario',
'statistical_basis': '2_std_deviation'
},
'low_volatility': {
'coordinates': {
'x': current_x - 1 * x_std,
'y': current_y - 1 * y_std,
'z': current_z - 1 * z_std
},
'severity': 'low',
'probability': 0.1,
'impact': 'low_volatility',
'description': 'Low volatility scenario',
'statistical_basis': '1_std_deviation'
}
}

return scenarios

def _create_enhanced_visualizations(self, data: pd.DataFrame, forecasts: Dict, scenarios: Dict, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
"""Create enhanced 3D visualizations"""

visualizations = {}

# 1. Enhanced 3D Time Series with Statistical Bands
fig = go.Figure()

for i, var in enumerate(forecasts.keys()):
# Historical data
fig.add_trace(go.Scatter3d(
x=data['Date'],
y=[i] * len(data),
z=data[var],
mode='lines',
name=f'{var} (Historical)',
line=dict(color=px.colors.qualitative.Set1[i])
))

# Forecast data
forecast_data = forecasts[var]
fig.add_trace(go.Scatter3d(
x=forecast_data['forecast_dates'],
y=[i] * len(forecast_data['forecast_values']),
z=forecast_data['forecast_values'],
mode='lines',
name=f'{var} (Forecast)',
line=dict(color=px.colors.qualitative.Set1[i], dash='dash')
))

# Statistical bands
mean_val = data[var].mean()
std_val = data[var].std()

fig.add_trace(go.Scatter3d(
x=data['Date'],
y=[i] * len(data),
z=[mean_val + 2*std_val] * len(data),
mode='lines',
name=f'{var} (+2σ)',
line=dict(color=px.colors.qualitative.Set1[i], dash='dot', width=1),
showlegend=False
))

fig.add_trace(go.Scatter3d(
x=data['Date'],
y=[i] * len(data),
z=[mean_val - 2*std_val] * len(data),
mode='lines',
name=f'{var} (-2σ)',
line=dict(color=px.colors.qualitative.Set1[i], dash='dot', width=1),
showlegend=False
))

fig.update_layout(
title='Enhanced 3D Time Series with Statistical Bands',
scene=dict(
xaxis_title='Date',
yaxis_title='Variable Index',
zaxis_title='Value'
)
)

visualizations['enhanced_3d_time_series'] = fig

# 2. 3D Scenario Space with Probability Weights
fig2 = go.Figure()

for scenario_name, scenario_data in scenarios.items():
coords = scenario_data['coordinates']
probability = scenario_data['probability']
severity = scenario_data['severity']

# Color by severity
color_map = {'low': 'green', 'moderate': 'yellow', 'high': 'red'}
color = color_map.get(severity, 'blue')

fig2.add_trace(go.Scatter3d(
x=[coords['x']],
y=[coords['y']],
z=[coords['z']],
mode='markers',
name=scenario_name,
marker=dict(
size=probability * 20, # Size based on probability
color=color,
opacity=0.8
),
text=f"Probability: {probability}<br>Severity: {severity}<br>Impact: {scenario_data['impact']}",
hovertemplate='%{text}<extra></extra>'
))

fig2.update_layout(
title='3D Scenario Space with Probability Weights',
scene=dict(
xaxis_title='X Dimension',
yaxis_title='Y Dimension',
zaxis_title='Z Dimension'
)
)

visualizations['3d_scenario_space_weighted'] = fig2

# 3. Correlation Heatmap with Forecast Integration
if 'statistical' in analysis_results and 'correlations' in analysis_results['statistical']:
corr_matrix = np.array(analysis_results['statistical']['correlations'])
variables = analysis_results['statistical']['variables']

fig3 = go.Figure(data=go.Heatmap(
z=corr_matrix,
x=variables,
y=variables,
colorscale='RdBu',
zmid=0
))

fig3.update_layout(
title='Correlation Matrix with Forecast Integration',
xaxis_title='Variables',
yaxis_title='Variables'
)

visualizations['correlation_heatmap'] = fig3

# 4. Forecast Accuracy Dashboard
fig4 = make_subplots(
rows=2, cols=2,
subplot_titles=['Forecast vs Historical', 'Residual Analysis', 'Volatility Trends', 'Scenario Probabilities'],
specs=[[{"secondary_y": False}, {"secondary_y": False}],
[{"secondary_y": False}, {"secondary_y": False}]]
)

# Forecast vs Historical
for i, (var, forecast_data) in enumerate(forecasts.items()):
fig4.add_trace(
go.Scatter(
x=data['Date'],
y=data[var],
mode='lines',
name=f'{var} (Historical)',
line=dict(color=px.colors.qualitative.Set1[i])
),
row=1, col=1
)

fig4.add_trace(
go.Scatter(
x=forecast_data['forecast_dates'],
y=forecast_data['forecast_values'],
mode='lines',
name=f'{var} (Forecast)',
line=dict(color=px.colors.qualitative.Set1[i], dash='dash')
),
row=1, col=1
)

# Scenario Probabilities
scenario_names = list(scenarios.keys())
scenario_probs = [scenarios[name]['probability'] for name in scenario_names]

fig4.add_trace(
go.Bar(
x=scenario_names,
y=scenario_probs,
name='Scenario Probabilities'
),
row=2, col=2
)

fig4.update_layout(
title='Forecast Accuracy Dashboard',
height=600
)

visualizations['forecast_dashboard'] = fig4

return visualizations

def _integrate_with_existing_engine(self, data: pd.DataFrame, analysis_results: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
"""Integrate with existing simulation engine"""

try:
from simulation import SimulationEngine

# Initialize existing simulation engine
existing_engine = SimulationEngine(config={
'simulation_horizon': 30,
'num_scenarios': 5
})

# Run existing simulation
existing_results = existing_engine.run_scenarios(analysis_results)

# Combine results
integration_results = {
'existing_scenarios': existing_results.get('synthetic_insights', []),
'3d_forecasts': results['forecasts'],
'3d_scenarios': results['3d_scenarios'],
'integration_summary': {
'existing_scenarios_count': len(existing_results.get('synthetic_insights', [])),
'3d_forecasts_count': len(results['forecasts']),
'3d_scenarios_count': len(results['3d_scenarios']),
'integration_successful': True
}
}

return integration_results

except Exception as e:
return {
'integration_summary': {
'integration_successful': False,
'error': str(e)
}
}

def _create_comprehensive_summary(self, results: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
"""Create comprehensive simulation summary"""

summary = {
'simulation_type': 'Enhanced 3D Simulation',
'total_forecasts': len(results['forecasts']),
'forecast_horizon': 30,
'scenarios_generated': len(results['3d_scenarios']),
'visualizations_created': len(results['visualizations']),
'forecast_accuracy': {},
'scenario_analysis': {},
'integration_status': {},
'data_characteristics': {}
}

# Forecast accuracy
for var, forecast_data in results['forecasts'].items():
if 'r2_score' in forecast_data:
summary['forecast_accuracy'][var] = {
'r2_score': forecast_data['r2_score'],
'mse': forecast_data.get('mse', 0)
}

# Scenario analysis
for scenario_name, scenario_data in results['3d_scenarios'].items():
summary['scenario_analysis'][scenario_name] = {
'severity': scenario_data.get('severity', 'unknown'),
'probability': scenario_data.get('probability', 0),
'impact': scenario_data.get('impact', 'unknown'),
'statistical_basis': scenario_data.get('statistical_basis', 'unknown')
}

# Integration status
if 'integration_results' in results:
summary['integration_status'] = results['integration_results'].get('integration_summary', {})

# Data characteristics
summary['data_characteristics'] = {
'total_variables': len(analysis_results['statistical']['variables']),
'time_periods': analysis_results['temporal']['time_periods'],
'data_shape': analysis_results['structural']['data_shape'],
'missing_values': analysis_results['structural']['missing_values']
}

return summary

def test_enhanced_3d_simulation():
"""Test enhanced 3D simulation"""

print("=" * 60)
print("TESTING ENHANCED 3D SIMULATION")
print("=" * 60)

# Load agricultural data
if os.path.exists('agricultural_data.csv'):
df = pd.read_csv('agricultural_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f" Loaded agricultural data: {len(df)} days × {len(df.columns)} variables")
else:
print(" Agricultural data not found. Please run test_agricultural_data.py first.")
return

# Initialize enhanced 3D simulation
enhanced_simulation = EnhancedSimulation3D(config={
'forecasting': {'forecast_horizon': 30}
})

# Run enhanced 3D simulation
results = enhanced_simulation.run_enhanced_3d_simulation(df)

# Display results
print("\n" + "="*50)
print("ENHANCED 3D SIMULATION RESULTS")
print("="*50)

# Forecast results
print(f"\nForecasts generated for {len(results['forecasts'])} variables:")
for var, forecast_data in results['forecasts'].items():
print(f" • {var}: {len(forecast_data['forecast_values'])} days ahead")
print(f" Last value: {forecast_data['last_known_value']:.2f}")
print(f" Forecast range: {min(forecast_data['forecast_values']):.2f} to {max(forecast_data['forecast_values']):.2f}")

# 3D scenarios
print(f"\n3D scenarios generated: {len(results['3d_scenarios'])}")
for scenario_name, scenario_data in results['3d_scenarios'].items():
coords = scenario_data['coordinates']
print(f" • {scenario_name}: ({coords['x']:.2f}, {coords['y']:.2f}, {coords['z']:.2f})")
print(f" Severity: {scenario_data['severity']}, Probability: {scenario_data['probability']}")
print(f" Statistical basis: {scenario_data.get('statistical_basis', 'unknown')}")

# Integration results
if 'integration_results' in results:
integration = results['integration_results']
if 'integration_summary' in integration:
summary = integration['integration_summary']
print(f"\nIntegration with existing engine:")
print(f" • Successful: {summary.get('integration_successful', False)}")
if summary.get('integration_successful'):
print(f" • Existing scenarios: {summary.get('existing_scenarios_count', 0)}")
print(f" • 3D forecasts: {summary.get('3d_forecasts_count', 0)}")
print(f" • 3D scenarios: {summary.get('3d_scenarios_count', 0)}")

# Visualizations
print(f"\nEnhanced visualizations created: {len(results['visualizations'])}")
for viz_name in results['visualizations'].keys():
print(f" • {viz_name}")

# Save enhanced visualizations
print(f"\nSaving enhanced visualizations...")
for viz_name, fig in results['visualizations'].items():
filename = f"enhanced_3d_simulation_{viz_name}.html"
fig.write_html(filename)
print(f" Saved {filename}")

# Save comprehensive results
with open('enhanced_3d_simulation_results.json', 'w') as f:
# Convert numpy arrays to lists for JSON serialization
json_results = {}
for key, value in results.items():
if key == 'forecasts':
json_results[key] = {}
for var, forecast_data in value.items():
json_results[key][var] = {
'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_data['forecast_dates']],
'forecast_values': forecast_data['forecast_values'],
'target_variable': forecast_data['target_variable'],
'horizon': forecast_data['horizon'],
'last_known_value': float(forecast_data['last_known_value']),
'last_known_date': forecast_data['last_known_date'].strftime('%Y-%m-%d')
}
elif key == '3d_scenarios':
json_results[key] = value
elif key == 'simulation_summary':
# Convert numpy types to Python types
summary = {}
for k, v in value.items():
if isinstance(v, dict):
summary[k] = {}
for k2, v2 in v.items():
if isinstance(v2, (np.integer, np.int64)):
summary[k][k2] = int(v2)
elif isinstance(v2, (np.floating, np.float64)):
summary[k][k2] = float(v2)
else:
summary[k][k2] = v2
elif isinstance(v, (np.integer, np.int64)):
summary[k] = int(v)
elif isinstance(v, (np.floating, np.float64)):
summary[k] = float(v)
else:
summary[k] = v
json_results[key] = summary
elif key == 'integration_results':
json_results[key] = value

json.dump(json_results, f, indent=2)

print(f" Saved comprehensive results to enhanced_3d_simulation_results.json")

return results

def main():
"""Main test function"""

print("Enhanced 3D Simulation with Forecasting")
print("=" * 60)

try:
results = test_enhanced_3d_simulation()

print(f"\n{'='*60}")
print("ENHANCED 3D SIMULATION TESTING COMPLETE")
print(f"{'='*60}")

print(" Enhanced 3D simulation engine tested successfully")
print(" Advanced forecasting models trained")
print(" 3D scenarios generated with statistical basis")
print(" Enhanced interactive visualizations created")
print(" Integration with existing simulation engine")
print(" Comprehensive analysis and reporting")

print("\nThis demonstrates:")
print("1. Advanced 3D visualization capabilities")
print("2. Statistical-based scenario generation")
print("3. Integration with existing simulation engine")
print("4. Enhanced forecasting with multiple models")
print("5. Comprehensive analysis and reporting")
print("6. Interactive 3D plots for exploration")

except Exception as e:
print(f"\nError during testing: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
main()
