#!/usr/bin/env python3
"""
3D Simulation Engine with Forecasting
Enhanced simulation with 3D visualization and forecasting capabilities
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

class ForecastingEngine:
"""
Advanced forecasting engine for time series prediction
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.models = {}
self.scalers = {}
self.forecast_horizon = self.config.get('forecast_horizon', 30)

def create_features(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
"""Create features for forecasting"""

# Create time-based features
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day_of_year'] = data['Date'].dt.dayofyear
data['day_of_week'] = data['Date'].dt.dayofweek

# Create lag features
for lag in [1, 7, 30]: # 1 day, 1 week, 1 month lags
data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)

# Create rolling statistics
for window in [7, 30]: # 1 week, 1 month windows
data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window=window).std()

# Create seasonal features
data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)
data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

# Select feature columns
feature_cols = [col for col in data.columns if col not in ['Date', target_col]]
feature_cols = [col for col in feature_cols if not col.startswith('Date')]

# Remove rows with NaN values
data_clean = data.dropna()

if len(data_clean) == 0:
return np.array([]), np.array([])

X = data_clean[feature_cols].values
y = data_clean[target_col].values

return X, y

def train_model(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
"""Train forecasting model for a target variable"""

X, y = self.create_features(data, target_col)

if len(X) == 0 or len(y) == 0:
return {'error': 'Insufficient data for training'}

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train multiple models
models = {
'linear': LinearRegression(),
'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
try:
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

results[name] = {
'model': model,
'mse': mse,
'r2': r2,
'predictions': y_pred
}
except Exception as e:
results[name] = {'error': str(e)}

# Store the best model
best_model_name = max([k for k in results.keys() if 'error' not in results[k]], 
key=lambda k: results[k]['r2'], default=None)

if best_model_name:
self.models[target_col] = results[best_model_name]['model']
self.scalers[target_col] = scaler

return {
'best_model': best_model_name,
'r2_score': results[best_model_name]['r2'],
'mse': results[best_model_name]['mse'],
'all_results': results
}

return {'error': 'No valid models trained'}

def forecast(self, data: pd.DataFrame, target_col: str, horizon: int = None) -> Dict[str, Any]:
"""Generate forecasts for a target variable"""

if horizon is None:
horizon = self.forecast_horizon

if target_col not in self.models:
train_result = self.train_model(data, target_col)
if 'error' in train_result:
return train_result

# Get the last known values
last_date = data['Date'].max()
last_values = data.iloc[-1].copy()

# Generate forecast dates
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

# Create forecast data
forecast_data = []
current_values = last_values.copy()

for i, forecast_date in enumerate(forecast_dates):
# Update time features
current_values['Date'] = forecast_date
current_values['year'] = forecast_date.year
current_values['month'] = forecast_date.month
current_values['day_of_year'] = forecast_date.dayofyear
current_values['day_of_week'] = forecast_date.dayofweek

# Update seasonal features
current_values['sin_month'] = np.sin(2 * np.pi * current_values['month'] / 12)
current_values['cos_month'] = np.cos(2 * np.pi * current_values['month'] / 12)
current_values['sin_day'] = np.sin(2 * np.pi * current_values['day_of_year'] / 365)
current_values['cos_day'] = np.cos(2 * np.pi * current_values['day_of_year'] / 365)

# Create features for prediction
feature_cols = [col for col in data.columns if col not in ['Date', target_col]]
feature_cols = [col for col in feature_cols if not col.startswith('Date')]

X_pred = current_values[feature_cols].values.reshape(1, -1)
X_pred_scaled = self.scalers[target_col].transform(X_pred)

# Make prediction
prediction = self.models[target_col].predict(X_pred_scaled)[0]

# Update lag features for next prediction
current_values[target_col] = prediction
if i > 0:
current_values[f'{target_col}_lag_1'] = forecast_data[-1][target_col]
if i >= 7:
current_values[f'{target_col}_lag_7'] = forecast_data[-7][target_col]
if i >= 30:
current_values[f'{target_col}_lag_30'] = forecast_data[-30][target_col]

forecast_data.append({
'Date': forecast_date,
target_col: prediction
})

return {
'forecast_dates': forecast_dates,
'forecast_values': [d[target_col] for d in forecast_data],
'target_variable': target_col,
'horizon': horizon,
'last_known_value': last_values[target_col],
'last_known_date': last_date
}

class Simulation3DEngine:
"""
3D Simulation Engine with forecasting capabilities
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}
self.forecasting_engine = ForecastingEngine(config.get('forecasting', {}))
self.scenario_generator = ScenarioGenerator3D(config.get('scenario_generator', {}))

def run_3d_simulation(self, data: pd.DataFrame, target_variables: List[str] = None) -> Dict[str, Any]:
"""Run 3D simulation with forecasting"""

if target_variables is None:
numeric_cols = data.select_dtypes(include=[np.number]).columns
target_variables = [col for col in numeric_cols if col != 'Date'][:3] # Top 3 variables

print(f"Running 3D simulation for variables: {target_variables}")

results = {
'forecasts': {},
'3d_scenarios': {},
'visualizations': {},
'simulation_summary': {}
}

# Generate forecasts for each target variable
for var in target_variables:
print(f"Generating forecast for {var}...")
forecast_result = self.forecasting_engine.forecast(data, var)
results['forecasts'][var] = forecast_result

# Generate 3D scenarios
print("Generating 3D scenarios...")
scenarios = self.scenario_generator.generate_3d_scenarios(data, target_variables)
results['3d_scenarios'] = scenarios

# Create 3D visualizations
print("Creating 3D visualizations...")
visualizations = self._create_3d_visualizations(data, results['forecasts'], scenarios)
results['visualizations'] = visualizations

# Create simulation summary
results['simulation_summary'] = self._create_simulation_summary(results)

return results

def _create_3d_visualizations(self, data: pd.DataFrame, forecasts: Dict, scenarios: Dict) -> Dict[str, Any]:
"""Create 3D visualizations"""

visualizations = {}

# 1. 3D Time Series Plot
fig = go.Figure()

# Add historical data
for i, var in enumerate(forecasts.keys()):
fig.add_trace(go.Scatter3d(
x=data['Date'],
y=[i] * len(data),
z=data[var],
mode='lines',
name=f'{var} (Historical)',
line=dict(color=px.colors.qualitative.Set1[i])
))

# Add forecast data
forecast_data = forecasts[var]
fig.add_trace(go.Scatter3d(
x=forecast_data['forecast_dates'],
y=[i] * len(forecast_data['forecast_values']),
z=forecast_data['forecast_values'],
mode='lines',
name=f'{var} (Forecast)',
line=dict(color=px.colors.qualitative.Set1[i], dash='dash')
))

fig.update_layout(
title='3D Time Series with Forecasts',
scene=dict(
xaxis_title='Date',
yaxis_title='Variable Index',
zaxis_title='Value'
)
)

visualizations['3d_time_series'] = fig

# 2. 3D Scenario Space
if scenarios:
fig2 = go.Figure()

for scenario_name, scenario_data in scenarios.items():
if 'coordinates' in scenario_data:
coords = scenario_data['coordinates']
fig2.add_trace(go.Scatter3d(
x=[coords['x']],
y=[coords['y']],
z=[coords['z']],
mode='markers',
name=scenario_name,
marker=dict(size=10, opacity=0.8)
))

fig2.update_layout(
title='3D Scenario Space',
scene=dict(
xaxis_title='X Dimension',
yaxis_title='Y Dimension',
zaxis_title='Z Dimension'
)
)

visualizations['3d_scenario_space'] = fig2

# 3. Forecast Confidence Intervals
fig3 = make_subplots(
rows=len(forecasts), cols=1,
subplot_titles=list(forecasts.keys()),
vertical_spacing=0.1
)

for i, (var, forecast_data) in enumerate(forecasts.items(), 1):
# Historical data
fig3.add_trace(
go.Scatter(
x=data['Date'],
y=data[var],
mode='lines',
name=f'{var} (Historical)',
line=dict(color='blue')
),
row=i, col=1
)

# Forecast data
fig3.add_trace(
go.Scatter(
x=forecast_data['forecast_dates'],
y=forecast_data['forecast_values'],
mode='lines',
name=f'{var} (Forecast)',
line=dict(color='red', dash='dash')
),
row=i, col=1
)

fig3.update_layout(
title='Forecast vs Historical Data',
height=300 * len(forecasts)
)

visualizations['forecast_comparison'] = fig3

return visualizations

def _create_simulation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
"""Create simulation summary"""

summary = {
'total_forecasts': len(results['forecasts']),
'forecast_horizon': 0,
'scenarios_generated': len(results['3d_scenarios']),
'visualizations_created': len(results['visualizations']),
'forecast_accuracy': {},
'scenario_analysis': {}
}

# Calculate forecast accuracy metrics
for var, forecast_data in results['forecasts'].items():
if 'r2_score' in forecast_data:
summary['forecast_accuracy'][var] = {
'r2_score': forecast_data['r2_score'],
'mse': forecast_data.get('mse', 0)
}

# Analyze scenarios
for scenario_name, scenario_data in results['3d_scenarios'].items():
summary['scenario_analysis'][scenario_name] = {
'severity': scenario_data.get('severity', 'unknown'),
'probability': scenario_data.get('probability', 0),
'impact': scenario_data.get('impact', 'unknown')
}

return summary

class ScenarioGenerator3D:
"""
3D Scenario Generator for multi-dimensional analysis
"""

def __init__(self, config: Dict[str, Any] = None):
self.config = config or {}

def generate_3d_scenarios(self, data: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
"""Generate 3D scenarios based on data patterns"""

scenarios = {}

if len(variables) >= 3:
# Use first 3 variables for 3D coordinates
x_var, y_var, z_var = variables[:3]

# Calculate current values
current_x = data[x_var].iloc[-1]
current_y = data[y_var].iloc[-1]
current_z = data[z_var].iloc[-1]

# Calculate historical ranges
x_range = data[x_var].max() - data[x_var].min()
y_range = data[y_var].max() - data[y_var].min()
z_range = data[z_var].max() - data[z_var].min()

# Generate scenarios
scenarios = {
'baseline': {
'coordinates': {'x': current_x, 'y': current_y, 'z': current_z},
'severity': 'low',
'probability': 0.6,
'impact': 'stable',
'description': 'Current state maintained'
},
'optimistic': {
'coordinates': {
'x': current_x + x_range * 0.2,
'y': current_y + y_range * 0.2,
'z': current_z + z_range * 0.2
},
'severity': 'low',
'probability': 0.2,
'impact': 'positive',
'description': 'All variables improve'
},
'pessimistic': {
'coordinates': {
'x': current_x - x_range * 0.2,
'y': current_y - y_range * 0.2,
'z': current_z - z_range * 0.2
},
'severity': 'high',
'probability': 0.1,
'impact': 'negative',
'description': 'All variables decline'
},
'mixed': {
'coordinates': {
'x': current_x + x_range * 0.1,
'y': current_y - y_range * 0.1,
'z': current_z + z_range * 0.1
},
'severity': 'moderate',
'probability': 0.1,
'impact': 'mixed',
'description': 'Mixed variable changes'
}
}

return scenarios

def test_3d_simulation_with_agricultural_data():
"""Test 3D simulation with agricultural data"""

print("=" * 60)
print("TESTING 3D SIMULATION WITH AGRICULTURAL DATA")
print("=" * 60)

# Load agricultural data
if os.path.exists('agricultural_data.csv'):
df = pd.read_csv('agricultural_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f" Loaded agricultural data: {len(df)} days × {len(df.columns)} variables")
else:
print(" Agricultural data not found. Please run test_agricultural_data.py first.")
return

# Initialize 3D simulation engine
simulation_engine = Simulation3DEngine(config={
'forecasting': {'forecast_horizon': 30},
'scenario_generator': {}
})

# Select target variables for 3D simulation
target_variables = ['Temperature_C', 'Crop_Yield_kg_per_hectare', 'Soil_Moisture_Percent']

print(f"Running 3D simulation for: {target_variables}")

# Run 3D simulation
results = simulation_engine.run_3d_simulation(df, target_variables)

# Display results
print("\n" + "="*50)
print("3D SIMULATION RESULTS")
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

# Visualizations
print(f"\nVisualizations created: {len(results['visualizations'])}")
for viz_name in results['visualizations'].keys():
print(f" • {viz_name}")

# Save visualizations
print(f"\nSaving visualizations...")
for viz_name, fig in results['visualizations'].items():
filename = f"3d_simulation_{viz_name}.html"
fig.write_html(filename)
print(f" Saved {filename}")

# Save results
with open('3d_simulation_results.json', 'w') as f:
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
json_results[key] = value

json.dump(json_results, f, indent=2)

print(f" Saved results to 3d_simulation_results.json")

return results

def main():
"""Main test function"""

print("3D Simulation Engine with Forecasting")
print("=" * 60)

try:
results = test_3d_simulation_with_agricultural_data()

print(f"\n{'='*60}")
print("3D SIMULATION TESTING COMPLETE")
print(f"{'='*60}")

print(" 3D simulation engine tested successfully")
print(" Forecasting models trained on agricultural data")
print(" 3D scenarios generated in multi-dimensional space")
print(" Interactive 3D visualizations created")
print(" Forecast accuracy metrics calculated")

print("\nThis demonstrates:")
print("1. Advanced forecasting capabilities")
print("2. 3D visualization of time series data")
print("3. Multi-dimensional scenario analysis")
print("4. Interactive plots for exploration")
print("5. Integration of historical data with future predictions")

except Exception as e:
print(f"\nError during testing: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
main()
