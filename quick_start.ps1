# Quick Start Script - Run this to start everything
Write-Host "Starting SCARCITY Framework Dashboard..." -ForegroundColor Green

# Start backend
Start-Process python -ArgumentList "sf_hfe_v2/backend_service.py" -WindowStyle Minimized

# Wait
Start-Sleep 3

# Start dashboard  
Start-Process python -ArgumentList "sf_hfe_v2/dashboard/app.py" -WindowStyle Minimized

# Wait
Start-Sleep 3

# Open browser
Start-Process "http://localhost:8050"

Write-Host "Dashboard started! Opening http://localhost:8050" -ForegroundColor Green
