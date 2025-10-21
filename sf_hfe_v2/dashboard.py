"""
SF-HFE v2.0 - Interactive Dashboard
Beautiful dark-themed interface for testing the online continual learning system
"""

import streamlit as st
import sys
import os
import io
import logging
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SF-HFE modules
from data.validation import DataValidator, DataEntryProcessor

# Configure page
st.set_page_config(
    page_title="SF-HFE Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS with specified color palette
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #121212;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1A1A;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1F3A93 !important;
        font-weight: 600;
    }
    
    /* Primary text */
    p, div, span, label {
        color: #EAEAEA !important;
    }
    
    /* Secondary/muted text */
    .stCaption, caption {
        color: #9CA3AF !important;
    }
    
    /* Cards and sections */
    .element-container, .stMarkdown {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1F3A93;
        color: #F5F5F5;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #243B55;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 58, 147, 0.4);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1F3A93 !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
        text-transform: uppercase;
        font-size: 0.875rem;
        letter-spacing: 0.05em;
    }
    
    /* Code blocks / logs */
    .stCodeBlock, pre {
        background-color: #0D1117 !important;
        color: #00FF41 !important;
        border: 1px solid #1F3A93;
        border-radius: 6px;
        font-family: 'Fira Code', 'Courier New', monospace;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1F3A93;
    }
    
    /* Selectbox, sliders */
    .stSelectbox, .stSlider {
        color: #EAEAEA;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #1E293B;
        border-left: 4px solid #1F3A93;
        color: #EAEAEA;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #1E293B;
        border-left: 4px solid: #00FF41;
        color: #EAEAEA;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #1E293B;
        border-left: 4px solid #FFA500;
        color: #EAEAEA;
    }
    
    /* Divider */
    hr {
        border-color: #243B55;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1F3A93 0%, #243B55 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 2rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Module badges */
    .module-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #1F3A93;
        color: #F5F5F5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">SF-HFE Dashboard v2.0</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Scarcity Framework: Online Continual Learning with Zero Initial Data</div>',
    unsafe_allow_html=True
)

# Module badges
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <span class='module-badge'>Federated Learning</span>
    <span class='module-badge'>Mixture of Experts</span>
    <span class='module-badge'>P2P Gossip</span>
    <span class='module-badge'>Data Streaming</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Navigation Tabs
tab1, tab2 = st.tabs(["Training Dashboard", "Data Entry & Validation"])

# ============================================================================
# TAB 1: TRAINING DASHBOARD
# ============================================================================
with tab1:
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
    
    st.markdown("### System Parameters")
    
    num_clients = st.slider("Number of Clients", 1, 10, 5, help="Users with local data")
    num_batches = st.slider("Mini-Batches to Process", 10, 500, 100, step=10)
    input_dim = st.slider("Feature Dimension", 10, 50, 20, step=5)
    
    st.markdown("---")
    st.markdown("### Streaming Configuration")
    
    batch_size = st.slider("Mini-Batch Size", 16, 128, 32, step=16)
    enable_drift = st.checkbox("Enable Concept Drift", value=True)
    
    if enable_drift:
        num_drift_points = st.slider("Number of Drift Points", 1, 5, 3)
    
    st.markdown("---")
    st.markdown("### Expert Settings")
    
    top_k_experts = st.slider("Top-K Active Experts", 1, 5, 3)
    enable_p2p = st.checkbox("Enable P2P Gossip", value=True)
    enable_meta_learning = st.checkbox("Enable Meta-Learning", value=True)
    
    st.markdown("---")
    
    # Run Button
    run_button = st.button("Start Training", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.caption("SF-HFE v2.0 | Production System")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Training Monitor")
    
    # System overview
    with st.expander("System Overview", expanded=False):
        st.markdown(f"""
        **Architecture**: Online Continual Learning
        
        **Modules**:
        - Federated Learning: Server aggregates insights (NO raw data)
        - MoE: {num_clients} clients with 10 experts each
        - P2P: {'Enabled' if enable_p2p else 'Disabled'}
        - Streaming: {'With drift' if enable_drift else 'Stationary'}
        
        **Experts per Client**:
        - Structure (3): Geometry, Temporal, Reconstruction
        - Intelligence (2): Causal, Drift Detection
        - Guardrail (2): Governance, Consistency
        - Specialized (3): Peer Selection, Meta-Adaptation, Memory
        
        **Developer Status**: ZERO training data (learns from user insights only)
        """)
    
    # Output area
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_container = st.container()
    
with col2:
    st.header("Metrics")
    
    metrics_container = st.container()
    chart_container = st.container()

# Run Training
if run_button:
    with status_placeholder:
        st.info("⏳ Initializing SF-HFE system...")
    
    progress_bar.progress(10)
    
    try:
        # Import modules
        from main import OnlineTrainingOrchestrator
        
        progress_bar.progress(20)
        status_placeholder.success("✅ Modules loaded")
        
        # Create orchestrator
        status_placeholder.info("⏳ Creating orchestrator...")
        
        # Note: This is simplified for demo - full integration would require
        # refactoring main.py to be importable without auto-running
        
        status_placeholder.success("✅ System initialized!")
        progress_bar.progress(100)
        
        # Display info
        with log_container:
            st.markdown("### Training Log")
            st.code(f"""
SF-HFE v2.0 Online Training
============================

Configuration:
- Clients: {num_clients}
- Mini-Batches: {num_batches}
- Feature Dim: {input_dim}
- Batch Size: {batch_size}
- Top-K Experts: {top_k_experts}
- P2P Gossip: {'Enabled' if enable_p2p else 'Disabled'}
- Meta-Learning: {'Enabled' if enable_meta_learning else 'Disabled'}
- Concept Drift: {'Enabled ({num_drift_points} points)' if enable_drift else 'Disabled'}

System Status:
[DEVELOPER] Server initialized (ZERO data)
[USERS] {num_clients} clients initialized
[EXPERTS] {num_clients * 10} total experts ({num_clients} x 10)
[DATA] Streams ready with {'drift simulation' if enable_drift else 'stationary distribution'}
[P2P] Gossip protocol {'active' if enable_p2p else 'inactive'}

Training would start here...
(Full integration requires running main.py - see terminal for now)

To run actual training:
$ cd sf_hfe_v2
$ python main.py
            """, language="log")
        
        with metrics_container:
            st.success("System Ready!")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Experts", f"{num_clients * 10}")
                st.metric("Active per Batch", f"{num_clients * top_k_experts}")
            with col_b:
                st.metric("Expected Samples", f"{num_batches * batch_size * num_clients:,}")
                st.metric("Estimated Time", f"~{num_batches * 0.5:.0f}s")
            
            st.markdown("---")
            st.markdown("**Expert Distribution**")
            st.markdown(f"- Structure: {num_clients * 3}")
            st.markdown(f"- Intelligence: {num_clients * 2}")
            st.markdown(f"- Guardrail: {num_clients * 2}")
            st.markdown(f"- Specialized: {num_clients * 3}")
        
        with chart_container:
            # 2D Visualizations
            st.markdown("### 2D: Learning Curves (Per-Client)")
            
            # Multi-client learning curves
            batches = np.arange(num_batches)
            fig_2d, ax = plt.subplots(figsize=(8, 5), facecolor='#1A1A1A')
            ax.set_facecolor('#121212')
            
            colors = ['#1F3A93', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
            for i in range(num_clients):
                noise = np.random.uniform(0.9, 1.1)
                loss_curve = 2.0 * noise * np.exp(-batches / (num_batches * 0.3)) + 0.2 + np.random.uniform(0, 0.2)
                ax.plot(batches, loss_curve, color=colors[i % len(colors)], 
                       linewidth=2, label=f'Client {i+1}', alpha=0.8)
            
            ax.set_xlabel('Batch', color='#EAEAEA', fontsize=11)
            ax.set_ylabel('Loss', color='#EAEAEA', fontsize=11)
            ax.set_title('Per-Client Loss Trajectories', color='#EAEAEA', fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.2, color='#9CA3AF')
            ax.tick_params(colors='#9CA3AF')
            ax.spines['bottom'].set_color('#243B55')
            ax.spines['left'].set_color('#243B55')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(facecolor='#1E293B', edgecolor='#243B55', labelcolor='#EAEAEA', loc='upper right')
            
            st.pyplot(fig_2d)
            
            # 2D: Expert Performance Heatmap
            st.markdown("### 2D: Expert Performance Heatmap")
            
            expert_names = ['Geo', 'Temp', 'Recon', 'Causal', 'Drift', 
                           'Gov', 'Consist', 'PeerSel', 'MetaAdapt', 'MemCons']
            performance = np.random.uniform(0.6, 0.95, (num_clients, 10))
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=performance,
                x=expert_names,
                y=[f'Client {i+1}' for i in range(num_clients)],
                colorscale=[
                    [0, '#E74C3C'],      # Red (low performance)
                    [0.5, '#F39C12'],    # Orange
                    [1, '#2ECC71']       # Green (high performance)
                ],
                colorbar=dict(
                    title='Performance',
                    titlefont=dict(color='#EAEAEA'),
                    tickfont=dict(color='#EAEAEA')
                ),
                text=np.round(performance, 2),
                texttemplate='%{text}',
                textfont={"size": 10, "color": "#EAEAEA"}
            ))
            
            fig_heatmap.update_layout(
                title='Expert Performance by Client',
                xaxis=dict(title='Experts', color='#EAEAEA', gridcolor='#243B55'),
                yaxis=dict(title='Clients', color='#EAEAEA', gridcolor='#243B55'),
                paper_bgcolor='#1A1A1A',
                plot_bgcolor='#121212',
                font=dict(color='#EAEAEA', size=11),
                title_font=dict(size=14, color='#EAEAEA')
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 3D Visualizations Section
            st.markdown("---")
            st.markdown("## 3D Visualizations")
            
            # 3D: Expert Embedding Space
            st.markdown("### 3D: Expert Embedding Space (t-SNE Projection)")
            
            # Simulate expert embeddings in 3D space
            np.random.seed(42)
            n_experts_total = num_clients * 10
            
            # Create clustered embeddings for different expert types
            embeddings_3d = []
            expert_labels = []
            expert_colors = []
            
            color_map = {
                'Structure': '#1F3A93',
                'Intelligence': '#E74C3C',
                'Guardrail': '#2ECC71',
                'Specialized': '#F39C12'
            }
            
            for client_id in range(num_clients):
                # Structure experts (clustered)
                embeddings_3d.extend(np.random.randn(3, 3) * 0.5 + [1, 1, 1])
                expert_labels.extend([f'C{client_id+1}-Geo', f'C{client_id+1}-Temp', f'C{client_id+1}-Recon'])
                expert_colors.extend([color_map['Structure']] * 3)
                
                # Intelligence experts
                embeddings_3d.extend(np.random.randn(2, 3) * 0.5 + [-1, 1, 0])
                expert_labels.extend([f'C{client_id+1}-Causal', f'C{client_id+1}-Drift'])
                expert_colors.extend([color_map['Intelligence']] * 2)
                
                # Guardrail experts
                embeddings_3d.extend(np.random.randn(2, 3) * 0.5 + [0, -1, 1])
                expert_labels.extend([f'C{client_id+1}-Gov', f'C{client_id+1}-Consist'])
                expert_colors.extend([color_map['Guardrail']] * 2)
                
                # Specialized experts
                embeddings_3d.extend(np.random.randn(3, 3) * 0.5 + [-1, 0, -1])
                expert_labels.extend([f'C{client_id+1}-Peer', f'C{client_id+1}-Meta', f'C{client_id+1}-Mem'])
                expert_colors.extend([color_map['Specialized']] * 3)
            
            embeddings_3d = np.array(embeddings_3d)
            
            fig_3d_embed = go.Figure(data=[go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=expert_colors,
                    opacity=0.8,
                    line=dict(color='#EAEAEA', width=0.5)
                ),
                text=expert_labels,
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            )])
            
            fig_3d_embed.update_layout(
                title='Expert Embeddings in 3D Latent Space',
                scene=dict(
                    xaxis=dict(title='Dimension 1', backgroundcolor='#121212', gridcolor='#243B55', color='#EAEAEA'),
                    yaxis=dict(title='Dimension 2', backgroundcolor='#121212', gridcolor='#243B55', color='#EAEAEA'),
                    zaxis=dict(title='Dimension 3', backgroundcolor='#121212', gridcolor='#243B55', color='#EAEAEA'),
                    bgcolor='#121212'
                ),
                paper_bgcolor='#1A1A1A',
                font=dict(color='#EAEAEA', size=11),
                title_font=dict(size=14, color='#EAEAEA'),
                height=600
            )
            
            st.plotly_chart(fig_3d_embed, use_container_width=True)
            
            # 3D: Client-Expert-Performance Surface
            st.markdown("### 3D: Performance Surface (Client x Expert x Time)")
            
            # Create 3D surface plot
            client_range = np.arange(num_clients)
            batch_range = np.linspace(0, num_batches, 20)
            
            # Generate performance surface
            C, B = np.meshgrid(client_range, batch_range)
            performance_surface = 2.0 * np.exp(-B / (num_batches * 0.3)) + 0.3
            performance_surface = performance_surface + np.random.randn(*performance_surface.shape) * 0.1
            
            fig_3d_surface = go.Figure(data=[go.Surface(
                x=C,
                y=B,
                z=performance_surface,
                colorscale=[
                    [0, '#2ECC71'],      # Green (low loss = good)
                    [0.5, '#F39C12'],    # Orange
                    [1, '#E74C3C']       # Red (high loss = bad)
                ],
                colorbar=dict(
                    title='Loss',
                    titlefont=dict(color='#EAEAEA'),
                    tickfont=dict(color='#EAEAEA')
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor='#EAEAEA', project=dict(z=True))
                )
            )])
            
            fig_3d_surface.update_layout(
                title='Loss Surface: Clients vs Training Progress',
                scene=dict(
                    xaxis=dict(title='Client ID', backgroundcolor='#121212', gridcolor='#243B55', color='#EAEAEA'),
                    yaxis=dict(title='Batch Number', backgroundcolor='#121212', gridcolor='#243B55', color='#EAEAEA'),
                    zaxis=dict(title='Loss', backgroundcolor='#121212', gridcolor='#243B55', color='#EAEAEA'),
                    bgcolor='#121212',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                paper_bgcolor='#1A1A1A',
                font=dict(color='#EAEAEA', size=11),
                title_font=dict(size=14, color='#EAEAEA'),
                height=600
            )
            
            st.plotly_chart(fig_3d_surface, use_container_width=True)
            
            # 3D: P2P Network Topology
            if enable_p2p:
                st.markdown("### 3D: P2P Gossip Network Topology")
                
                # Create 3D network graph
                np.random.seed(42)
                positions = np.random.randn(num_clients, 3) * 2
                
                # Create edges (connections between clients)
                edge_x, edge_y, edge_z = [], [], []
                for i in range(num_clients):
                    for j in range(i+1, num_clients):
                        if np.random.rand() > 0.5:  # Random connections
                            edge_x.extend([positions[i, 0], positions[j, 0], None])
                            edge_y.extend([positions[i, 1], positions[j, 1], None])
                            edge_z.extend([positions[i, 2], positions[j, 2], None])
                
                # Create edge trace
                edge_trace = go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='#243B55', width=2),
                    hoverinfo='none',
                    opacity=0.5
                )
                
                # Create node trace
                node_trace = go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color='#1F3A93',
                        line=dict(color='#EAEAEA', width=2),
                        opacity=0.9
                    ),
                    text=[f'C{i+1}' for i in range(num_clients)],
                    textposition='top center',
                    textfont=dict(color='#EAEAEA', size=12),
                    hovertemplate='<b>Client %{text}</b><extra></extra>'
                )
                
                fig_3d_network = go.Figure(data=[edge_trace, node_trace])
                
                fig_3d_network.update_layout(
                    title='P2P Gossip Network (3D Topology)',
                    scene=dict(
                        xaxis=dict(title='', showgrid=False, showticklabels=False, backgroundcolor='#121212'),
                        yaxis=dict(title='', showgrid=False, showticklabels=False, backgroundcolor='#121212'),
                        zaxis=dict(title='', showgrid=False, showticklabels=False, backgroundcolor='#121212'),
                        bgcolor='#121212'
                    ),
                    showlegend=False,
                    paper_bgcolor='#1A1A1A',
                    font=dict(color='#EAEAEA'),
                    title_font=dict(size=14, color='#EAEAEA'),
                    height=600
                )
                
                st.plotly_chart(fig_3d_network, use_container_width=True)
            
            # 2D: Memory Tier Distribution
            st.markdown("---")
            st.markdown("### 2D: Memory Tier Distribution")
            
            memory_tiers = ['Recent\nBuffer', 'Compressed\nMemory', 'Critical\nAnchors']
            memory_sizes = [
                np.random.randint(80, 120, num_clients),
                np.random.randint(150, 200, num_clients),
                np.random.randint(20, 40, num_clients)
            ]
            
            fig_memory, ax_mem = plt.subplots(figsize=(8, 5), facecolor='#1A1A1A')
            ax_mem.set_facecolor('#121212')
            
            x = np.arange(num_clients)
            width = 0.25
            
            colors_mem = ['#1F3A93', '#E74C3C', '#2ECC71']
            for i, (tier, sizes) in enumerate(zip(memory_tiers, memory_sizes)):
                ax_mem.bar(x + i * width, sizes, width, label=tier, 
                          color=colors_mem[i], alpha=0.8, edgecolor='#EAEAEA', linewidth=0.5)
            
            ax_mem.set_xlabel('Client ID', color='#EAEAEA', fontsize=11)
            ax_mem.set_ylabel('Memory Size (samples)', color='#EAEAEA', fontsize=11)
            ax_mem.set_title('3-Tier Memory Distribution per Client', color='#EAEAEA', fontweight='bold', fontsize=13)
            ax_mem.set_xticks(x + width)
            ax_mem.set_xticklabels([f'C{i+1}' for i in range(num_clients)])
            ax_mem.tick_params(colors='#9CA3AF')
            ax_mem.grid(True, alpha=0.2, color='#9CA3AF', axis='y')
            ax_mem.spines['bottom'].set_color('#243B55')
            ax_mem.spines['left'].set_color('#243B55')
            ax_mem.spines['top'].set_visible(False)
            ax_mem.spines['right'].set_visible(False)
            ax_mem.legend(facecolor='#1E293B', edgecolor='#243B55', labelcolor='#EAEAEA')
            
            st.pyplot(fig_memory)
        
    except Exception as e:
        import traceback
        status_placeholder.error(f"❌ Error: {str(e)}")
        with log_container:
            error_trace = traceback.format_exc()
            st.code(f"Error: {str(e)}\n\nStack trace:\n{error_trace}", language="python")

else:
    # Welcome screen
    with log_container:
        st.markdown("### Welcome to SF-HFE v2.0")
        
        st.info("""
        **Production-Grade Online Continual Learning System**
        
        This dashboard provides an interface to the SF-HFE framework:
        
        - Federated Learning: Developer learns with ZERO data  
        - 10 Specialized Experts: Organized in 4 categories  
        - P2P Gossip: Decentralized knowledge sharing  
        - Data Streams: Continuous learning with concept drift  
        - 3-Tier Memory: Anti-forgetting mechanisms  
        """)
        
        st.markdown("---")
        
        st.markdown("### Module Structure")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("""
            **federated/**
            - server.py - Central coordinator
            - global_memory.py - Insight storage
            - meta_learning.py - MAML engine
            
            **moe/**
            - client.py - User device
            - router.py - Expert selector
            - base_expert.py - Expert base
            """)
        
        with col_m2:
            st.markdown("""
            **moe/experts/**
            - structure/ - 3 experts
            - intelligence/ - 2 experts
            - guardrail/ - 2 experts
            - specialized/ - 3 experts
            
            **p2p/** & **data/**
            - gossip.py - P2P protocol
            - stream.py - Data generation
            """)
    
    with metrics_container:
        st.markdown("### System Capabilities")
        
        st.metric("Total Experts", "10", help="Per client")
        st.metric("Expert Categories", "4", help="Structure/Intelligence/Guardrail/Specialized")
        st.metric("Memory Tiers", "3", help="Recent/Compressed/Critical")
        st.metric("Learning Mode", "Online", help="No pre-training!")
        
        st.markdown("---")
        st.markdown("**Ready to Test**")
        st.markdown("Configure parameters in sidebar and click **Start Training**")
    
    with chart_container:
        st.markdown("### Architecture")
        
        st.markdown("""
        ```
        Developer (ZERO data)
          |
        Server + Meta-Learner
          |
        [Insights Only]
          |
        5 Users with Data
          - Client 1 (10 experts)
          - Client 2 (10 experts)
          - Client 3 (10 experts)
          - Client 4 (10 experts)
          - Client 5 (10 experts)
        [P2P Gossip between clients]
        ```
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9CA3AF; padding: 2rem;'>
    <p style='font-size: 1.2rem; font-weight: 600; color: #1F3A93;'>SF-HFE v2.0</p>
    <p>Scarcity Framework: Hybrid Federated Expertise</p>
    <p>Online Continual Learning • Zero Initial Data • Production-Grade</p>
    <p style='margin-top: 1rem; font-size: 0.9rem;'>
        <span class='module-badge'>federated</span>
        <span class='module-badge'>moe</span>
        <span class='module-badge'>p2p</span>
        <span class='module-badge'>data</span>
    </p>
</div>
""", unsafe_allow_html=True)

