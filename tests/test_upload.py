#!/usr/bin/env python3
"""
Test script to upload data to the SCARCITY Framework backend
"""

import requests
import pandas as pd
import numpy as np
import time

def create_test_data():
"""Create test economic data"""
np.random.seed(42)
n_samples = 100

data = {
'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
'price': 100 + np.cumsum(np.random.normal(0, 2, n_samples)),
'volume': np.random.randint(1000, 10000, n_samples)
}

df = pd.DataFrame(data)
return df

def test_upload():
"""Test uploading data to the backend"""
print("Creating test data...")
df = create_test_data()

print(f"Created test data: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Save to CSV
csv_path = "test_data.csv"
df.to_csv(csv_path, index=False)
print(f"Saved test data to {csv_path}")

# Upload to backend
print("Uploading data to backend...")
url = "http://127.0.0.1:5000/api/upload"

with open(csv_path, 'rb') as f:
files = {'file': ('test_data.csv', f, 'text/csv')}
response = requests.post(url, files=files)

print(f"Upload response: {response.status_code}")
if response.status_code == 200:
print("Upload successful!")
result = response.json()
print(f"Result: {result}")

# Monitor processing
print("Monitoring processing...")
for i in range(30): # Monitor for 30 seconds
status_url = "http://127.0.0.1:5000/api/status"
status_response = requests.get(status_url)

if status_response.status_code == 200:
status = status_response.json()
print(f"Status {i+1}: {status.get('current_step', 'Unknown')} - {status.get('progress', 0)}%")

if status.get('is_processing', False) == False:
print("Processing completed!")
break
else:
print(f"Status check failed: {status_response.status_code}")

time.sleep(1)

# Get insights
print("Getting insights...")
insights_url = "http://127.0.0.1:5000/api/insights"
insights_response = requests.get(insights_url)

if insights_response.status_code == 200:
insights = insights_response.json()
print(f"Insights: {insights}")
else:
print(f"Insights request failed: {insights_response.status_code}")

else:
print(f"Upload failed: {response.status_code}")
print(f"Response: {response.text}")

if __name__ == "__main__":
test_upload()
