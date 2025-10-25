"""
Test script to upload Market_Prices.csv and test the system
"""

import pandas as pd
import requests
import json
import os

def test_system():
"""Test the complete system with Market_Prices.csv"""

print("Testing SCARCITY Framework Dashboard System...")

# 1. Check if backend is running
try:
response = requests.get("http://localhost:5000/api/status")
if response.status_code == 200:
print("[OK] Backend service is running")
print(f" Status: {response.json()}")
else:
print("[ERROR] Backend service not responding")
return False
except Exception as e:
print(f"[ERROR] Cannot connect to backend: {e}")
return False

# 2. Check if dashboard is running
try:
response = requests.get("http://localhost:8050")
if response.status_code == 200:
print("[OK] Dashboard is running")
else:
print("[ERROR] Dashboard not responding")
return False
except Exception as e:
print(f"[ERROR] Cannot connect to dashboard: {e}")
return False

# 3. Upload Market_Prices.csv
csv_path = "Market_Prices.csv"
if not os.path.exists(csv_path):
print(f"[ERROR] Market_Prices.csv not found at {csv_path}")
return False

try:
print(f"[UPLOAD] Uploading {csv_path}...")
with open(csv_path, 'rb') as f:
files = {'file': f}
response = requests.post("http://localhost:5000/api/upload", files=files)

if response.status_code == 200:
result = response.json()
print("[OK] File uploaded successfully")
print(f" Result: {result}")
else:
print(f"[ERROR] Upload failed: {response.text}")
return False

except Exception as e:
print(f"[ERROR] Upload error: {e}")
return False

# 4. Test insights generation
try:
print("[TEST] Testing insights generation...")
response = requests.post(
"http://localhost:5000/api/generate-insights",
json={"type": "full"}
)

if response.status_code == 200:
result = response.json()
print("[OK] Insights generated successfully")
print(f" Result: {result}")
else:
print(f"[ERROR] Insights generation failed: {response.text}")

except Exception as e:
print(f"[ERROR] Insights generation error: {e}")

# 5. Test insights retrieval
try:
print("[TEST] Testing insights retrieval...")
response = requests.get("http://localhost:5000/api/insights")

if response.status_code == 200:
result = response.json()
print("[OK] Insights retrieved successfully")
print(f" Result: {result}")
else:
print(f"[ERROR] Insights retrieval failed: {response.text}")

except Exception as e:
print(f"[ERROR] Insights retrieval error: {e}")

print("\n[SUCCESS] System test completed!")
print("You can now access the dashboard at: http://localhost:8050")
return True

if __name__ == "__main__":
test_system()