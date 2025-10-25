#!/usr/bin/env python3
"""
Quick Agricultural Data Analysis
Shows the agricultural data characteristics and patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_agricultural_data():
"""Analyze the synthetic agricultural data"""

print("=" * 60)
print("AGRICULTURAL DATA ANALYSIS")
print("=" * 60)

# Load the agricultural data
df = pd.read_csv('agricultural_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Dataset: {len(df)} days × {len(df.columns)} variables")
print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

# Show variable statistics
print(f"\nVariable Statistics:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
stats = df[numeric_cols].describe()
print(stats.round(2))

# Show correlations
print(f"\nCorrelation Matrix:")
corr_matrix = df[numeric_cols].corr()
print(corr_matrix.round(3))

# Show seasonal patterns
print(f"\nSeasonal Patterns:")
df['Month'] = df['Date'].dt.month
monthly_stats = df.groupby('Month')[['Temperature_C', 'Humidity_Percent', 'Crop_Yield_kg_per_hectare']].mean()
print(monthly_stats.round(2))

# Show key insights
print(f"\nKey Agricultural Insights:")

# Temperature range
temp_range = df['Temperature_C'].max() - df['Temperature_C'].min()
print(f"• Temperature range: {df['Temperature_C'].min():.1f}°C to {df['Temperature_C'].max():.1f}°C (range: {temp_range:.1f}°C)")

# Crop yield statistics
yield_mean = df['Crop_Yield_kg_per_hectare'].mean()
yield_std = df['Crop_Yield_kg_per_hectare'].std()
print(f"• Average crop yield: {yield_mean:.0f} kg/hectare (±{yield_std:.0f})")

# Rainfall patterns
rainy_days = (df['Rainfall_mm'] > 0).sum()
total_rainfall = df['Rainfall_mm'].sum()
print(f"• Rainy days: {rainy_days} out of {len(df)} days ({rainy_days/len(df)*100:.1f}%)")
print(f"• Total rainfall: {total_rainfall:.0f} mm over {len(df)} days")

# Pest infestation
pest_mean = df['Pest_Infestation_Level'].mean()
pest_max = df['Pest_Infestation_Level'].max()
print(f"• Average pest infestation: {pest_mean:.3f} (max: {pest_max:.3f})")

# Fertilizer usage
fertilizer_days = (df['Fertilizer_Usage_kg_per_hectare'] > 0).sum()
total_fertilizer = df['Fertilizer_Usage_kg_per_hectare'].sum()
print(f"• Fertilizer application days: {fertilizer_days} out of {len(df)} days")
print(f"• Total fertilizer used: {total_fertilizer:.0f} kg/hectare")

# Irrigation
irrigation_days = (df['Irrigation_Hours'] > 0).sum()
total_irrigation = df['Irrigation_Hours'].sum()
print(f"• Irrigation days: {irrigation_days} out of {len(df)} days")
print(f"• Total irrigation: {total_irrigation:.0f} hours")

# Correlations with crop yield
print(f"\nCrop Yield Correlations:")
yield_corr = df[numeric_cols].corr()['Crop_Yield_kg_per_hectare'].sort_values(ascending=False)
for var, corr in yield_corr.items():
if var != 'Crop_Yield_kg_per_hectare':
print(f"• {var}: {corr:.3f}")

return df

def main():
"""Main analysis function"""

try:
df = analyze_agricultural_data()

print(f"\n{'='*60}")
print("AGRICULTURAL DATA ANALYSIS COMPLETE")
print(f"{'='*60}")

print(" Agricultural data successfully analyzed")
print(" Seasonal patterns identified")
print(" Correlations calculated")
print(" Key insights extracted")

print("\nThis demonstrates:")
print("1. Realistic agricultural data patterns")
print("2. Seasonal variations in temperature, humidity, and yield")
print("3. Correlations between environmental factors and crop yield")
print("4. Pest infestation and management patterns")
print("5. Irrigation and fertilizer usage patterns")

except Exception as e:
print(f"\nError during analysis: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
main()
