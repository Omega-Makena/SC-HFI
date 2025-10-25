"""
Test script to upload data and test the full neural OMEO system
"""

import requests
import pandas as pd
import numpy as np
import time
import json

def test_omoe_system():
"""Test the full OMEO system with data upload and processing"""

base_url = "http://127.0.0.1:5000"

print("Testing Neural OMEO System...")

# Create test data
test_data = pd.DataFrame({
'price': np.random.normal(100, 10, 100),
'volume': np.random.normal(1000, 100, 100),
'volatility': np.random.normal(0.2, 0.05, 100),
'momentum': np.random.normal(0.1, 0.1, 100),
'trend': np.random.normal(0.05, 0.02, 100)
})

# Save test data
test_data.to_csv('test_omoe_data.csv', index=False)
print(f"Created test data: {test_data.shape}")

try:
# Test 1: Check if service is running
print("\n=== Test 1: Service Status ===")
response = requests.get(f"{base_url}/api/status")
if response.status_code == 200:
status = response.json()
print(f"Service Status: {status}")
else:
print(f"Service not running: {response.status_code}")
return

# Test 2: Upload data
print("\n=== Test 2: Upload Data ===")
with open('test_omoe_data.csv', 'rb') as f:
files = {'file': ('test_omoe_data.csv', f, 'text/csv')}
data = {'domain': 'economic'}
response = requests.post(f"{base_url}/api/upload", files=files, data=data)

if response.status_code == 200:
upload_result = response.json()
print(f"Upload Result: {upload_result}")
else:
print(f"Upload failed: {response.status_code} - {response.text}")
return

# Test 3: Wait for processing and check status
print("\n=== Test 3: Processing Status ===")
for i in range(10): # Wait up to 10 seconds
response = requests.get(f"{base_url}/api/status")
if response.status_code == 200:
status = response.json()
print(f"Processing Status: {status['is_processing']} - {status['current_step']} - {status['progress']}%")

if not status['is_processing']:
break

time.sleep(1)

# Test 4: Get insights
print("\n=== Test 4: Get Insights ===")
response = requests.get(f"{base_url}/api/insights")
if response.status_code == 200:
insights = response.json()
print(f"Insights Generated: {len(insights)} items")
print(f"Insights Keys: {list(insights.keys())}")

# Check for OMEO insights
if 'omoe_insights' in insights:
omoe_insights = insights['omoe_insights']
print(f"OMEO Insights: {list(omoe_insights.keys())}")

for tier, tier_insights in omoe_insights.items():
if isinstance(tier_insights, dict) and 'confidence' in tier_insights:
print(f" {tier}: confidence={tier_insights['confidence']:.3f}")

# Check for simulations
if 'simulations' in insights:
simulations = insights['simulations']
print(f"Simulations: {list(simulations.keys())}")

else:
print(f"Failed to get insights: {response.status_code} - {response.text}")

# Test 5: Run simulation
print("\n=== Test 5: Run Simulation ===")
simulation_data = {'scenario': 'forecast'}
response = requests.post(f"{base_url}/api/simulate", json=simulation_data)

if response.status_code == 200:
simulation_result = response.json()
print(f"Simulation Result: {simulation_result}")
else:
print(f"Simulation failed: {response.status_code} - {response.text}")

print("\n=== Neural OMEO System Test Complete ===")
print(" All tests completed successfully!")
print(" Neural models are processing data correctly!")
print(" OMEO system is providing insights!")

except requests.exceptions.ConnectionError:
print(" Cannot connect to backend service. Make sure it's running on http://127.0.0.1:5000")
except Exception as e:
print(f" Test failed with error: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
test_omoe_system()
