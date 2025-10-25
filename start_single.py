"""
Single Command SCARCITY Framework Dashboard Startup
Starts both services and manages them as a single unit
"""

import subprocess
import sys
import time
import webbrowser
import signal
import os
from pathlib import Path

class DashboardManager:
def __init__(self):
self.processes = []
self.running = True

def start_backend(self):
"""Start backend service"""
print("Starting SCARCITY Framework Backend Service...")
backend_path = Path(__file__).parent / "sf_hfe_v2" / "backend_service.py"

try:
process = subprocess.Popen([
sys.executable, str(backend_path)
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

self.processes.append(process)
print(f"Backend started with PID: {process.pid}")
return process

except Exception as e:
print(f"Error starting backend: {e}")
return None

def start_dashboard(self):
"""Start dashboard service"""
print("Starting Dashboard Frontend...")
dashboard_path = Path(__file__).parent / "sf_hfe_v2" / "dashboard" / "app.py"

try:
process = subprocess.Popen([
sys.executable, str(dashboard_path)
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

self.processes.append(process)
print(f"Dashboard started with PID: {process.pid}")
return process

except Exception as e:
print(f"Error starting dashboard: {e}")
return None

def signal_handler(self, signum, frame):
"""Handle shutdown signals"""
print("\n\nShutting down services...")
self.running = False
self.stop_all()
sys.exit(0)

def stop_all(self):
"""Stop all processes"""
for process in self.processes:
if process.poll() is None: # Process is still running
print(f"Stopping process {process.pid}...")
process.terminate()
try:
process.wait(timeout=5)
except subprocess.TimeoutExpired:
print(f"Force killing process {process.pid}...")
process.kill()

print("All services stopped.")

def check_services(self):
"""Check if services are responding"""
import requests

# Check backend
try:
response = requests.get("http://localhost:5000/api/status", timeout=5)
if response.status_code == 200:
print("[OK] Backend service is responding")
return True
except:
print("[WARNING] Backend service not accessible")
return False

def run(self):
"""Main run method"""
print("=" * 60)
print("SCARCITY Framework Dashboard - Single Command Startup")
print("=" * 60)

# Set up signal handlers
signal.signal(signal.SIGINT, self.signal_handler)
signal.signal(signal.SIGTERM, self.signal_handler)

# Start backend
backend_process = self.start_backend()
if not backend_process:
print("Failed to start backend. Exiting.")
return

# Wait for backend
print("Waiting for backend to initialize...")
time.sleep(5)

# Start dashboard
dashboard_process = self.start_dashboard()
if not dashboard_process:
print("Failed to start dashboard. Exiting.")
self.stop_all()
return

# Wait for dashboard
print("Waiting for dashboard to initialize...")
time.sleep(5)

# Check services
print("\nChecking service status...")
if self.check_services():
print("[OK] Backend service is ready")

# Open browser
print("\nOpening dashboard in browser...")
try:
webbrowser.open("http://localhost:8050")
print("Dashboard opened in your default browser")
except:
print("Could not open browser automatically")
print("Please open http://localhost:8050 in your browser")

print("\n" + "=" * 60)
print("SYSTEM READY!")
print("=" * 60)
print("Backend API: http://localhost:5000")
print("Dashboard: http://localhost:8050")
print("=" * 60)
print("\nBoth services are running as managed processes.")
print("Press Ctrl+C to stop all services")

# Monitor processes
try:
while self.running:
time.sleep(1)

# Check if processes are still running
for i, process in enumerate(self.processes):
if process.poll() is not None:
print(f"Process {process.pid} stopped unexpectedly")
self.running = False
break

except KeyboardInterrupt:
self.signal_handler(signal.SIGINT, None)

def main():
"""Main function"""
manager = DashboardManager()
manager.run()

if __name__ == "__main__":
main()
