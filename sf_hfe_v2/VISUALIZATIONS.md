# SF-HFE Dashboard Visualizations

## Overview
The SF-HFE v2.0 dashboard includes comprehensive 2D and 3D visualizations to monitor and analyze the online continual learning system.

---

## 2D Visualizations

### 1. Learning Curves (Per-Client)
**Type**: Line plot  
**Purpose**: Track loss trajectories for each client over training batches  
**Features**:
- Multi-colored lines (one per client)
- Shows convergence patterns
- Identifies struggling clients
- Dark theme compatible

### 2. Expert Performance Heatmap
**Type**: Interactive heatmap (Plotly)  
**Purpose**: Compare expert performance across all clients  
**Features**:
- 10 experts (columns) × N clients (rows)
- Color-coded: Red (poor) → Orange → Green (excellent)
- Hover to see exact performance values
- Identifies best/worst performing experts per client

### 3. Memory Tier Distribution
**Type**: Grouped bar chart  
**Purpose**: Visualize 3-tier memory usage per client  
**Features**:
- Recent Buffer (blue)
- Compressed Memory (red)
- Critical Anchors (green)
- Shows memory balance across clients

---

## 3D Visualizations

### 1. Expert Embedding Space (t-SNE Projection)
**Type**: 3D scatter plot (Plotly)  
**Purpose**: Visualize expert specialization in latent space  
**Features**:
- Interactive rotation/zoom
- Color-coded by expert category:
  - Structure: Navy Blue (#1F3A93)
  - Intelligence: Red (#E74C3C)
  - Guardrail: Green (#2ECC71)
  - Specialized: Orange (#F39C12)
- Hover shows expert name and coordinates
- Reveals clustering patterns

**Interpretation**:
- Tight clusters = similar expert behaviors
- Distant points = specialized experts
- Cross-category mixing = versatile learning

### 2. Performance Surface (Client × Time)
**Type**: 3D surface plot (Plotly)  
**Purpose**: Show loss evolution across clients and training time  
**Features**:
- X-axis: Client ID
- Y-axis: Batch number (time)
- Z-axis: Loss value
- Color gradient: Green (low loss) → Red (high loss)
- Contour projections on floor
- Rotatable camera

**Interpretation**:
- Flat surface = consistent performance
- Valleys = good learning
- Peaks = struggling clients/batches
- Smooth gradients = stable learning

### 3. P2P Gossip Network Topology
**Type**: 3D network graph (Plotly)  
**Purpose**: Visualize peer-to-peer connections between clients  
**Features**:
- Nodes = Clients (labeled C1, C2, ...)
- Edges = P2P connections
- Interactive 3D rotation
- Shows decentralized communication structure

**Interpretation**:
- Dense connections = high gossip activity
- Isolated nodes = poor connectivity
- Central nodes = hub clients
- Topology affects knowledge diffusion

---

## Usage Tips

### Interactive Features (3D Plots)
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Shift + drag
- **Reset**: Double-click
- **Hover**: See detailed tooltips

### Dark Theme Compatibility
All visualizations use the SF-HFE color palette:
- Background: Charcoal (#121212)
- Accents: Navy Blue (#1F3A93)
- Text: Off-white (#EAEAEA)
- Grids: Cool Grey (#9CA3AF)

---

## Technical Details

### Dependencies
- `matplotlib`: 2D static plots
- `plotly`: Interactive 3D visualizations
- `numpy`: Data generation

### Performance
- Visualizations are generated on-demand
- 3D plots use WebGL for smooth rendering
- Optimized for 5-10 clients (scalable to more)

---

## Customization

### Adding New Visualizations

1. **2D Plot (Matplotlib)**:
```python
fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1A1A1A')
ax.set_facecolor('#121212')
# ... plot your data ...
ax.tick_params(colors='#9CA3AF')
st.pyplot(fig)
```

2. **3D Plot (Plotly)**:
```python
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
fig.update_layout(
    scene=dict(bgcolor='#121212'),
    paper_bgcolor='#1A1A1A',
    font=dict(color='#EAEAEA')
)
st.plotly_chart(fig, use_container_width=True)
```

---

## Future Enhancements

Planned visualizations:
- Real-time loss streaming
- Expert attention weights (heatmap)
- Concept drift detection timeline
- Meta-learning parameter evolution
- Fairness metrics dashboard (HFI)
- Client contribution analysis

---

**Enjoy exploring your SF-HFE system in 2D and 3D!**

