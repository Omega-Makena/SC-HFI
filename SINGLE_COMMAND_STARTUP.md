# SCARCITY Framework Dashboard - Single Command Startup

## **UNIFIED STARTUP OPTIONS**

### **Option 1: Windows Batch File (EASIEST)**
```cmd
START_DASHBOARD.bat
```
**What it does:**
- Starts backend in a new window
- Starts dashboard in a new window 
- Opens browser automatically
- Shows status messages

### **Option 2: Python Single Command**
```bash
python start.py
```
**What it does:**
- Starts both services in separate windows
- Opens browser automatically
- Monitors service status
- Handles shutdown gracefully

### **Option 3: PowerShell Script**
```powershell
powershell -ExecutionPolicy Bypass -File start_dashboard.ps1
```

### **Option 4: Manual (Two Commands)**
```bash
# Terminal 1:
cd sf_hfe_v2 && python backend_service.py

# Terminal 2: 
cd sf_hfe_v2/dashboard && python app.py
```

## **RECOMMENDED: Use START_DASHBOARD.bat**

**Why this is the best option:**
- **Single click** - Just double-click the file
- **Separate windows** - Each service runs in its own window
- **Auto browser** - Opens dashboard automatically
- **Clear status** - Shows what's happening
- **Easy to stop** - Just close the windows
- **No dependencies** - Works on any Windows system

## 🌐 **Access Points**

- **Dashboard:** http://localhost:8050
- **Backend API:** http://localhost:5000

## **What Happens When You Run START_DASHBOARD.bat:**

1. **Backend Service** starts in a new window (port 5000)
2. **Dashboard Service** starts in a new window (port 8050)
3. **Browser** opens automatically to the dashboard
4. **Status** shows both services are ready
5. **Ready** for data upload and analysis!

## **Test Your Setup:**

After starting, test with:
```bash
python test_system.py
```

## 🆘 **Troubleshooting:**

- **Services not starting:** Check if ports 5000/8050 are free
- **Browser not opening:** Manually go to http://localhost:8050
- **Permission errors:** Run as Administrator if needed
- **Python not found:** Ensure Python is in your PATH

---

## 🎉 **READY TO GO!**

**Just double-click `START_DASHBOARD.bat` and you're ready to analyze data with the SCARCITY Framework!**

