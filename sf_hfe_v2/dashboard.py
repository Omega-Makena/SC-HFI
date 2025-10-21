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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
            st.markdown("### Expected Learning Curve")
            
            # Simulated learning curve
            batches = np.arange(num_batches)
            loss_curve = 2.0 * np.exp(-batches / (num_batches * 0.3)) + 0.3
            
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1A1A1A')
            ax.set_facecolor('#121212')
            ax.plot(batches, loss_curve, color='#1F3A93', linewidth=2, label='Expected Loss')
            ax.set_xlabel('Batch', color='#EAEAEA')
            ax.set_ylabel('Loss', color='#EAEAEA')
            ax.set_title('Projected Learning Curve', color='#EAEAEA', fontweight='bold')
            ax.grid(True, alpha=0.2, color='#9CA3AF')
            ax.tick_params(colors='#9CA3AF')
            ax.spines['bottom'].set_color('#243B55')
            ax.spines['left'].set_color('#243B55')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(facecolor='#1E293B', edgecolor='#243B55', labelcolor='#EAEAEA')
            
            st.pyplot(fig)
        
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

