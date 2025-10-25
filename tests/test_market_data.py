"""
Test OMEO system with real market data
"""

import requests
import pandas as pd
import json

def test_omoe_with_market_data():
"""Test the OMEO system with Market_Prices.csv"""
base_url = "http://127.0.0.1:5000"

print("Testing OMEO System with Market_Prices.csv...")

# Test 1: Check service status
try:
response = requests.get(f"{base_url}/api/status")
if response.status_code == 200:
status = response.json()
print(f"Service Status: {status}")
else:
print(f"Service not responding: {response.status_code}")
return
except Exception as e:
print(f"Cannot connect to service: {e}")
return

# Test 2: Upload market data
try:
with open('Market_Prices.csv', 'rb') as f:
files = {'file': ('Market_Prices.csv', f, 'text/csv')}
response = requests.post(f"{base_url}/api/upload", files=files)

if response.status_code == 200:
upload_result = response.json()
print(f"Upload Result: {upload_result}")
else:
print(f"Upload failed: {response.status_code} - {response.text}")
return
except Exception as e:
print(f"Upload error: {e}")
return

# Test 3: Monitor processing
print("\n=== Monitoring Processing ===")
for i in range(10):
try:
response = requests.get(f"{base_url}/api/status")
if response.status_code == 200:
status = response.json()
progress = status.get('progress', 0)
current_step = status.get('current_step', 'Unknown')
is_processing = status.get('is_processing', False)

print(f"Processing Status: {is_processing} - {current_step} - {progress}%")

if not is_processing and progress == 100:
print("Processing completed!")
break

except Exception as e:
print(f"Status check error: {e}")
break

# Test 4: Get insights
try:
response = requests.get(f"{base_url}/api/insights")
if response.status_code == 200:
insights = response.json()
print(f"\nInsights Generated: {len(insights)} items")
print(f"Insights Keys: {list(insights.keys())}")

# Show OMEO insights
if 'omoe_insights' in insights:
omoe_insights = insights['omoe_insights']
print(f"\nOMEO Insights:")
for tier_name, tier_data in omoe_insights.items():
confidence = tier_data.get('confidence', 'N/A')
print(f" {tier_name}: confidence={confidence}")

# Show simulations
if 'simulations' in insights:
simulations = insights['simulations']
print(f"\nSimulations: {list(simulations.keys())}")

else:
print(f"Insights failed: {response.status_code} - {response.text}")
except Exception as e:
print(f"Insights error: {e}")

# Test 5: Run simulation
try:
simulation_data = {
"scenario": "Market Forecast",
"parameters": {
"forecast_horizon": "30 days",
"confidence_level": 0.95
}
}

response = requests.post(f"{base_url}/api/simulate", json=simulation_data)
if response.status_code == 200:
simulation_result = response.json()
print(f"\nSimulation Result:")
print(f"Scenario: {simulation_result.get('scenario', 'N/A')}")
print(f"Results: {simulation_result.get('results', {})}")
else:
print(f"Simulation failed: {response.status_code} - {response.text}")
except Exception as e:
print(f"Simulation error: {e}")

print("\n=== Market Data OMEO Test Complete ===")

if __name__ == "__main__":
test_omoe_with_market_data()
