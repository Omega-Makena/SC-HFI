# SCARCITY Framework Dashboard - Quick Start Guide

## **AUTOMATIC STARTUP OPTIONS**

### **Option 1: PowerShell Script (Recommended)**
```powershell
powershell -ExecutionPolicy Bypass -File quick_start.ps1
```

### **Option 2: Batch File**
```cmd
start_dashboard.bat
```

### **Option 3: Python Script**
```bash
python start_dashboard_auto.py
```

### **Option 4: Manual Start**
```bash
# Terminal 1: Backend
cd sf_hfe_v2
python backend_service.py

# Terminal 2: Dashboard
cd sf_hfe_v2/dashboard 
python app.py
```

## 🌐 **Access Points**

- **Dashboard:** http://localhost:8050
- **Backend API:** http://localhost:5000

## **What You Can Do**

1. **Upload Data:** Use the "Upload New Data" button to upload CSV files
2. **Run Analysis:** Click "Run Analysis" to process your data
3. **View Results:** See real-time metrics and charts
4. **Export Data:** Download analysis results

## **System Requirements**

- Python 3.8+
- Required packages: dash, plotly, pandas, requests, flask

## **File Structure**

```
scarcity/
├── start_dashboard_auto.py # Python startup script
├── start_dashboard.bat # Windows batch file
├── start_dashboard.ps1 # PowerShell script
├── quick_start.ps1 # Quick PowerShell start
├── test_system.py # System test script
├── Market_Prices.csv # Sample data file
└── sf_hfe_v2/
├── backend_service.py # Backend service
└── dashboard/ # Dashboard frontend
├── app.py # Main dashboard app
├── pages/ # Dashboard pages
├── components/ # UI components
└── utils/ # Utilities
```

## **Test Results**

The system has been tested with Market_Prices.csv (5,515 rows) and is working perfectly:

- Backend service running on port 5000
- Dashboard running on port 8050 
- File upload working
- Real-time data updates
- Error handling implemented
- No hardcoded values

## **Quick Test**

Run this to test everything:
```bash
python test_system.py
```

## 🆘 **Troubleshooting**

- **Port conflicts:** Make sure ports 5000 and 8050 are available
- **Python not found:** Ensure Python is in your PATH
- **Permission errors:** Run PowerShell as Administrator if needed
- **Browser not opening:** Manually go to http://localhost:8050

---

**Ready to analyze your data with the SCARCITY Framework!** 🎉
