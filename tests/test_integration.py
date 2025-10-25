"""
SCARCITY Framework Dashboard - Single Process Integration Test
Runs both backend and frontend in one process for testing
"""

import threading
import time
import webbrowser
import sys
import os
from pathlib import Path

def run_backend():
"""Run backend service in a thread"""
try:
# Add sf_hfe_v2 to Python path
sf_hfe_path = Path(__file__).parent / "sf_hfe_v2"
sys.path.insert(0, str(sf_hfe_path))

# Import and run backend
from backend_service import app as backend_app

print("Starting Backend Service on port 5000...")
backend_app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

except Exception as e:
print(f"Backend error: {e}")

def run_dashboard():
"""Run dashboard service in a thread"""
try:
# Add dashboard to Python path
dashboard_path = Path(__file__).parent / "sf_hfe_v2" / "dashboard"
sys.path.insert(0, str(dashboard_path))

# Import and run dashboard
from app import create_dashboard_app

print("Starting Dashboard Service on port 8050...")
app = create_dashboard_app()
app.run(host='127.0.0.1', port=8050, debug=False, use_reloader=False)

except Exception as e:
print(f"Dashboard error: {e}")

def test_integration():
"""Test the integration between services"""
import requests
import time

print("\nTesting Integration...")

# Wait for services to start
time.sleep(8)

# Test backend
try:
response = requests.get("http://localhost:5000/api/status", timeout=5)
if response.status_code == 200:
print(" Backend API is responding")
else:
print(" Backend API not responding")
except Exception as e:
print(f" Backend test failed: {e}")

# Test dashboard
try:
response = requests.get("http://localhost:8050", timeout=5)
if response.status_code == 200:
print(" Dashboard is responding")
else:
print(" Dashboard not responding")
except Exception as e:
print(f" Dashboard test failed: {e}")

# Test file upload
try:
print("\nTesting file upload...")
csv_path = Path(__file__).parent / "Market_Prices.csv"
if csv_path.exists():
with open(csv_path, 'rb') as f:
files = {'file': f}
response = requests.post("http://localhost:5000/api/upload", files=files, timeout=10)

if response.status_code == 200:
result = response.json()
print(f" File upload successful: {result.get('rows', 0)} rows uploaded")
else:
print(f" File upload failed: {response.text}")
else:
print("⚠️ Market_Prices.csv not found for upload test")
except Exception as e:
print(f" Upload test failed: {e}")

# Test insights generation
try:
print("\nTesting insights generation...")
response = requests.post(
"http://localhost:5000/api/generate-insights",
json={"type": "full"},
timeout=10
)

if response.status_code == 200:
result = response.json()
print(f" Insights generation: {result}")
else:
print(f" Insights generation failed: {response.text}")
except Exception as e:
print(f" Insights test failed: {e}")

def main():
"""Main function"""
print("=" * 60)
print("SCARCITY Framework - Single Process Integration Test")
print("=" * 60)

# Start backend in a thread
backend_thread = threading.Thread(target=run_backend, daemon=True)
backend_thread.start()

# Wait a moment
time.sleep(2)

# Start dashboard in a thread
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

# Wait a moment
time.sleep(2)

# Open browser
print("\nOpening Dashboard in Browser...")
try:
webbrowser.open("http://localhost:8050")
print(" Browser opened")
except:
print("⚠️ Could not open browser automatically")
print(" Please open http://localhost:8050 manually")

# Run integration tests
test_integration()

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE!")
print("=" * 60)
print("Backend API: http://localhost:5000")
print("Dashboard: http://localhost:8050")
print("=" * 60)
print("\nBoth services are running in the same process.")
print("You can now test the full integration!")
print("Press Ctrl+C to stop all services.")

try:
# Keep running
while True:
time.sleep(1)

# Check if threads are still alive
if not backend_thread.is_alive():
print(" Backend thread stopped")
break

if not dashboard_thread.is_alive():
print(" Dashboard thread stopped")
break

except KeyboardInterrupt:
print("\n\nStopping all services...")
print(" All services stopped")

if __name__ == "__main__":
main()

