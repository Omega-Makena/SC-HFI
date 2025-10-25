"""
Quick Integration Test Script
Tests the SCARCITY Framework Dashboard integration
"""

import requests
import time
import os
from pathlib import Path

def test_backend():
"""Test backend service"""
print("Testing Backend Service...")
try:
response = requests.get("http://localhost:5000/api/status", timeout=5)
if response.status_code == 200:
print("[OK] Backend is running")
return True
else:
print("[ERROR] Backend not responding")
return False
except Exception as e:
print(f"[ERROR] Backend test failed: {e}")
return False

def test_dashboard():
"""Test dashboard service"""
print("Testing Dashboard Service...")
try:
response = requests.get("http://localhost:8050", timeout=5)
if response.status_code == 200:
print("[OK] Dashboard is running")
return True
else:
print("[ERROR] Dashboard not responding")
return False
except Exception as e:
print(f"[ERROR] Dashboard test failed: {e}")
return False

def test_file_upload():
"""Test file upload functionality"""
print("Testing File Upload...")
csv_path = Path(__file__).parent / "Market_Prices.csv"

if not csv_path.exists():
print("[WARNING] Market_Prices.csv not found")
return False

try:
with open(csv_path, 'rb') as f:
files = {'file': f}
response = requests.post("http://localhost:5000/api/upload", files=files, timeout=10)

if response.status_code == 200:
result = response.json()
print(f"[OK] File upload successful: {result.get('rows', 0)} rows")
return True
else:
print(f"[ERROR] Upload failed: {response.text}")
return False
except Exception as e:
print(f"[ERROR] Upload test failed: {e}")
return False

def test_insights():
"""Test insights generation"""
print("Testing Insights Generation...")
try:
response = requests.post(
"http://localhost:5000/api/generate-insights",
json={"type": "full"},
timeout=10
)

if response.status_code == 200:
result = response.json()
print(f"[OK] Insights generated: {result}")
return True
else:
print(f"[ERROR] Insights failed: {response.text}")
return False
except Exception as e:
print(f"[ERROR] Insights test failed: {e}")
return False

def main():
"""Main test function"""
print("=" * 50)
print("SCARCITY Framework Integration Test")
print("=" * 50)

# Wait for services to start
print("Waiting for services to start...")
time.sleep(5)

# Run tests
tests = [
("Backend Service", test_backend),
("Dashboard Service", test_dashboard),
("File Upload", test_file_upload),
("Insights Generation", test_insights)
]

results = []
for test_name, test_func in tests:
print(f"\n--- {test_name} ---")
result = test_func()
results.append((test_name, result))

# Summary
print("\n" + "=" * 50)
print("TEST RESULTS SUMMARY")
print("=" * 50)

passed = 0
for test_name, result in results:
status = "[PASS]" if result else "[FAIL]"
print(f"{test_name}: {status}")
if result:
passed += 1

print(f"\nPassed: {passed}/{len(results)} tests")

if passed == len(results):
print("\n[SUCCESS] ALL TESTS PASSED! Integration is working perfectly!")
print("You can now use the dashboard at: http://localhost:8050")
else:
print(f"\n[WARNING] {len(results) - passed} tests failed. Check the services.")

if __name__ == "__main__":
main()
