"""
Ultra-Simple SCARCITY Framework Dashboard Startup
One command to rule them all!
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def main():
print("Starting SCARCITY Framework Dashboard...")
print("=" * 50)

# Get the current directory
current_dir = Path(__file__).parent

# Start backend
print("Starting Backend Service...")
backend_cmd = [
sys.executable, 
str(current_dir / "sf_hfe_v2" / "backend_service.py")
]

backend_process = subprocess.Popen(
backend_cmd,
creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
)

print(f"Backend started (PID: {backend_process.pid})")

# Wait for backend
print("Waiting for backend to initialize...")
time.sleep(5)

# Start dashboard
print("Starting Dashboard Frontend...")
dashboard_cmd = [
sys.executable,
str(current_dir / "sf_hfe_v2" / "dashboard" / "app.py")
]

dashboard_process = subprocess.Popen(
dashboard_cmd,
creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
)

print(f"Dashboard started (PID: {dashboard_process.pid})")

# Wait for dashboard
print("Waiting for dashboard to initialize...")
time.sleep(5)

# Open browser
print("Opening Dashboard in Browser...")
try:
webbrowser.open("http://localhost:8050")
print("Browser opened successfully")
except:
print("Could not open browser automatically")
print("Please open http://localhost:8050 manually")

# Success message
print("\n" + "=" * 50)
print("SCARCITY Framework Dashboard is READY!")
print("=" * 50)
print(f"Backend API: http://localhost:5000")
print(f"Dashboard: http://localhost:8050")
print("=" * 50)

print("\nBoth services are running in separate windows.")
print("Close those windows to stop the services.")
print("Or press Ctrl+C here to exit this launcher.")

try:
# Keep the launcher running
while True:
time.sleep(1)

# Check if processes are still running
if backend_process.poll() is not None:
print("Backend process stopped")
break

if dashboard_process.poll() is not None:
print("Dashboard process stopped")
break

except KeyboardInterrupt:
print("\nGoodbye! Services are still running in their windows.")
print("Close those windows to stop the services completely.")

if __name__ == "__main__":
main()