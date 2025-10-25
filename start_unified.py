"""
Unified SCARCITY Framework Dashboard Startup
Runs both backend and frontend in a single process with threading
"""

import threading
import time
import webbrowser
import sys
import os
from pathlib import Path

def run_backend():
"""Run the backend service in a separate thread"""
try:
# Import and run backend
backend_path = Path(__file__).parent / "sf_hfe_v2" / "backend_service.py"

# Add the sf_hfe_v2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "sf_hfe_v2"))

# Import and run the backend
from backend_service import app as backend_app

print("Starting SCARCITY Framework Backend Service...")
backend_app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

except Exception as e:
print(f"Error starting backend: {e}")

def run_dashboard():
"""Run the dashboard service in a separate thread"""
try:
# Import and run dashboard
dashboard_path = Path(__file__).parent / "sf_hfe_v2" / "dashboard"
sys.path.insert(0, str(dashboard_path))

from app import create_dashboard_app

print("Starting Dashboard Frontend...")
app = create_dashboard_app()
app.run(host='127.0.0.1', port=8050, debug=False, use_reloader=False)

except Exception as e:
print(f"Error starting dashboard: {e}")

def main():
"""Main function to start both services"""
print("=" * 60)
print("SCARCITY Framework Dashboard - Unified Startup")
print("=" * 60)

# Start backend in a separate thread
backend_thread = threading.Thread(target=run_backend, daemon=True)
backend_thread.start()

# Wait for backend to start
print("Waiting for backend to initialize...")
time.sleep(5)

# Start dashboard in a separate thread
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

# Wait for dashboard to start
print("Waiting for dashboard to initialize...")
time.sleep(5)

# Open browser
print("Opening dashboard in browser...")
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
print("\nBoth services are running in the same process.")
print("Press Ctrl+C to stop all services")

try:
# Keep the main thread alive
while True:
time.sleep(1)

# Check if threads are still alive
if not backend_thread.is_alive():
print("Backend thread stopped")
break

if not dashboard_thread.is_alive():
print("Dashboard thread stopped")
break

except KeyboardInterrupt:
print("\n\nShutting down services...")
print("All services stopped. Goodbye!")

if __name__ == "__main__":
main()
