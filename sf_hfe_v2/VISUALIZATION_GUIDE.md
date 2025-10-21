# SF-HFE v2.0 - Test Visualization Guide

## Generated Visualizations

After running `test_domain_clients.py`, the following visualizations are automatically generated in the `test_visualizations/` directory:

---

## 2D Visualizations (PNG, High-Resolution)

### 1. Learning Curves (`1_learning_curves.png`)
**Type**: Line plot (300 DPI)  
**Purpose**: Track loss trajectories for all three domains over training time

**Features**:
- Multi-colored lines (Agriculture: Blue, Tech: Red, Economics: Green)
- Shows convergence patterns
- Identifies struggling domains
- Dark theme (#121212 background)

**Interpretation**:
- Downward trend = Learning progress
- Flat lines = Convergence or plateau
- Spikes = Concept drift or difficult batches

---

### 2. Expert Usage Distribution (`2_expert_usage.png`)
**Type**: Stacked bar chart (300 DPI)  
**Purpose**: Visualize which experts each domain used

**Features**:
- 100% stacked bars (one per domain)
- 10 colors (one per expert type)
- Shows expert specialization per domain
- Legend with all 10 experts

**Interpretation**:
- Tall sections = Frequently used experts
- Dominant colors = Core experts for each domain
- Small sections = Rarely used specialized experts

---

### 3. Per-Expert Comparison (`3_per_expert_comparison.png`)
**Type**: Grouped bar chart (300 DPI)  
**Purpose**: Compare usage of each expert type across all domains

**Features**:
- 10 expert groups (X-axis)
- 3 bars per group (one per domain)
- Direct cross-domain comparison
- Rotated X-labels for readability

**Interpretation**:
- Tall bars = Popular expert for that domain
- Similar heights = Expert used equally across domains
- One tall bar = Domain-specific expert

---

## 3D Visualizations (HTML, Interactive)

### 4. 3D Loss Surface (`4_3d_loss_surface.html`)
**Type**: Interactive 3D surface plot (Plotly)  
**Purpose**: Visualize loss evolution across domains and time

**Features**:
- X-axis: Domain (Agriculture, Tech, Economics)
- Y-axis: Batch number (time)
- Z-axis: Loss value
- Color gradient: Green (low) → Orange → Red (high)
- Contour projections on floor
- Fully rotatable (click + drag)

**Interaction**:
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Shift + drag
- **Hover**: See exact values

**Interpretation**:
- Valleys = Good learning regions
- Peaks = High loss (struggling)
- Smooth surfaces = Consistent performance
- Sharp changes = Concept drift or events

---

### 5. 3D Expert Usage (`5_3d_expert_usage.html`)
**Type**: Interactive 3D scatter plot (Plotly)  
**Purpose**: Visualize expert usage patterns in 3D space

**Features**:
- X-axis: Domain ID
- Y-axis: Expert ID
- Z-axis: Usage percentage
- Color intensity: Viridis colormap
- Marker size: 10px
- Hover tooltips with expert names

**Interaction**:
- Rotate to see from different angles
- Hover over points for details
- Toggle domains on/off (legend)

**Interpretation**:
- High points = Frequently used experts
- Clusters = Similar usage patterns
- Isolated points = Domain-specific experts

---

### 6. 3D Training Trajectories (`6_3d_training_trajectory.html`)
**Type**: Interactive 3D line plot (Plotly)  
**Purpose**: Show learning paths for each domain in 3D space

**Features**:
- X-axis: Domain ID
- Y-axis: Batch number (time)
- Z-axis: Loss value
- Colored lines (Agriculture: Blue, Tech: Red, Economics: Green)
- Markers at each batch
- Line thickness: 4px

**Interaction**:
- Rotate to see trajectories from different angles
- Hover for exact batch/loss values
- Click legend to hide/show domains

**Interpretation**:
- Descending lines = Learning progress
- Smooth curves = Stable learning
- Jagged lines = Noisy or unstable training
- Compare slopes to see which domain learns fastest

---

## File Specifications

### 2D PNG Files
- **Resolution**: 300 DPI (print-quality)
- **Format**: PNG with transparency
- **Background**: Dark theme (#1A1A1A)
- **Text color**: Off-white (#EAEAEA)
- **Accent colors**: Navy Blue (#1F3A93), Red (#E74C3C), Green (#2ECC71)
- **File size**: ~200-400 KB each

### 3D HTML Files
- **Format**: Standalone HTML (self-contained)
- **Library**: Plotly.js (embedded)
- **Interactivity**: Full 3D rotation, zoom, pan, hover
- **Theme**: Dark (#1A1A1A background, #EAEAEA text)
- **File size**: ~1-2 MB each (includes Plotly library)
- **Browser**: Works in any modern browser (Chrome, Firefox, Edge, Safari)

---

## Viewing the Visualizations

### 2D Images
Open with any image viewer:
```bash
# Windows
start test_visualizations/1_learning_curves.png

# Mac/Linux
open test_visualizations/1_learning_curves.png
```

### 3D Interactive Plots
Open with any web browser:
```bash
# Windows
start test_visualizations/4_3d_loss_surface.html

# Mac/Linux
open test_visualizations/4_3d_loss_surface.html
```

Or simply double-click the HTML files!

---

## Customization

To modify visualizations, edit `test_domain_clients.py`:

**Change colors**:
```python
colors_2d = ['#YOUR_COLOR1', '#YOUR_COLOR2', '#YOUR_COLOR3']
```

**Change DPI (resolution)**:
```python
plt.savefig(..., dpi=300)  # Increase for higher quality
```

**Change figure size**:
```python
fig, ax = plt.subplots(figsize=(12, 6))  # Width, Height in inches
```

**Change 3D camera angle**:
```python
camera=dict(eye=dict(x=1.7, y=1.7, z=1.4))  # Adjust x, y, z
```

---

## Sample Output

After a successful test run, you'll see:

```
================================================================================
GENERATING VISUALIZATIONS
================================================================================

Creating 2D visualizations...
  ✓ Saved: test_visualizations/1_learning_curves.png
  ✓ Saved: test_visualizations/2_expert_usage.png
  ✓ Saved: test_visualizations/3_per_expert_comparison.png

Creating 3D visualizations...
  ✓ Saved: test_visualizations/4_3d_loss_surface.html
  ✓ Saved: test_visualizations/5_3d_expert_usage.html
  ✓ Saved: test_visualizations/6_3d_training_trajectory.html

================================================================================
All visualizations saved to: test_visualizations/
================================================================================
```

---

## Interpretation Tips

### For Loss Curves:
- Look for downward trends (good)
- Compare convergence rates between domains
- Identify which domain struggles most

### For Expert Usage:
- Structure experts (Geo, Temp, Recon) typically dominate early training
- Specialized experts (Meta, Memory) activate later
- Different domains may prefer different experts

### For 3D Surfaces:
- Green valleys = Successful learning regions
- Red peaks = Areas needing more training
- Smooth gradients = Stable learning dynamics

---

## Troubleshooting

**No visualizations generated?**
- Check that matplotlib is installed: `pip install matplotlib`
- Check that plotly is installed: `pip install plotly`
- Verify `test_visualizations/` directory was created

**Images look blurry?**
- Increase DPI in the code (default is 300)

**3D plots don't load?**
- Make sure you're opening HTML files in a browser
- Check browser console for errors (F12)
- Try a different browser

**Unicode errors in console?**
- These are Windows encoding issues (harmless)
- Visualizations still save correctly

---

**All visualizations use the SF-HFE dark theme for consistency!**

