"""
Scarcity Framework - Interactive Web Dashboard

A Streamlit-based frontend for testing and visualizing all stages of the framework.
"""

import streamlit as st
import sys
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import numpy as np

# Import the framework
from scarcity.run import (
    run_stage1_simulation,
    run_federated_learning,
    run_insight_exchange,
    run_expert_routing,
    run_meta_learning,
    run_p2p_gossip,
    run_structure_discovery
)

# Configure page
st.set_page_config(
    page_title="Scarcity Framework Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .metric-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        text-align: center;
    }
    .log-box {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Scarcity Framework Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Interactive Testing Interface for Federated Learning with Expert Systems**")
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stage Selection
    stage = st.selectbox(
        "Select Stage",
        [
            "Stage 1 - Local Training & Insights",
            "Stage 2 - Federated Learning (FedAvg)",
            "Stage 3 - Insight Exchange",
            "Stage 4 - Expert Routing",
            "Stage 5 - Meta-Learning (Reptile)",
            "Stage 6 - P2P Gossip",
            "Stage 8 - Self-Supervised Structure Discovery"
        ],
        index=6  # Default to Stage 8
    )
    
    st.markdown("---")
    
    # Hyperparameters
    st.subheader("Hyperparameters")
    
    if "Stage 1" in stage:
        num_clients = st.slider("Number of Clients", 2, 10, 3)
        num_rounds = 1
        input_dim = 10
        output_dim = 1
    else:
        num_clients = st.slider("Number of Clients", 2, 20, 5)
        num_rounds = st.slider("Number of Rounds", 1, 10, 5)
        input_dim = st.slider("Input Dimension", 5, 50, 10, step=5)
        output_dim = st.slider("Output Dimension", 1, 10, 1)
    
    data_size = st.slider("Data Size per Client", 50, 500, 100, step=50)
    
    st.markdown("---")
    
    # Expert Settings (for Stage 4+)
    if any(s in stage for s in ["Stage 4", "Stage 5", "Stage 6", "Stage 8"]):
        st.subheader("Expert Settings")
        router_strategy = st.selectbox(
            "Router Strategy",
            ["variance", "random"],
            index=0
        )
        epochs = st.slider("Training Epochs", 1, 20, 5)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
    
    st.markdown("---")
    
    # Run Button
    run_button = st.button("🚀 Run Experiment", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"📊 {stage}")
    
    # Stage Description
    stage_descriptions = {
        "Stage 1": "Simple local training with fake data. Each client trains experts and generates insights.",
        "Stage 2": "Federated Learning with FedAvg aggregation. Clients train, send weights to server, receive global model.",
        "Stage 3": "Scarcity-style Insight Exchange. Clients send structured insights instead of raw weights.",
        "Stage 4": "Expert Routing Architecture. Router selects which expert to train based on data characteristics.",
        "Stage 5": "Reptile-style Meta-Learning. Server computes and broadcasts global meta-parameters.",
        "Stage 6": "P2P Gossip Mechanism. Clients exchange expert weights directly with peers.",
        "Stage 8": "Self-Supervised Structure Discovery. All experts train with autoencoder objectives on unlabeled data."
    }
    
    for key, desc in stage_descriptions.items():
        if key in stage:
            st.info(f"ℹ️ **{desc}**")
            break
    
    # Output Area
    output_container = st.container()
    
with col2:
    st.header("📈 Metrics")
    metrics_container = st.container()

# Run Experiment
if run_button:
    with output_container:
        st.subheader("📝 Execution Log")
        
        # Create a string buffer to capture logs
        log_capture = io.StringIO()
        
        # Setup custom logging handler
        class StreamlitLogHandler(logging.Handler):
            def __init__(self, stream):
                super().__init__()
                self.stream = stream
                
            def emit(self, record):
                msg = self.format(record)
                self.stream.write(msg + '\n')
        
        # Configure logging
        logger = logging.getLogger()
        handler = StreamlitLogHandler(log_capture)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("⏳ Initializing...")
            progress_bar.progress(10)
            
            # Run the selected stage
            if "Stage 1" in stage:
                status_text.text("⏳ Running Stage 1...")
                progress_bar.progress(30)
                insights = run_stage1_simulation(num_clients=num_clients)
                result_data = {"insights": len(insights)}
                
            elif "Stage 2" in stage:
                status_text.text("⏳ Running Stage 2...")
                progress_bar.progress(30)
                server, clients, _ = run_federated_learning(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                result_data = {"clients": len(clients), "rounds": num_rounds}
                
            elif "Stage 3" in stage:
                status_text.text("⏳ Running Stage 3...")
                progress_bar.progress(30)
                server, clients, meta_learner = run_insight_exchange(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    input_dim=input_dim
                )
                result_data = {
                    "clients": len(clients),
                    "rounds": num_rounds,
                    "insights": len(meta_learner.insight_memory)
                }
                
            elif "Stage 4" in stage:
                status_text.text("⏳ Running Stage 4...")
                progress_bar.progress(30)
                server, clients, meta_learner = run_expert_routing(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                result_data = {
                    "clients": len(clients),
                    "experts_per_client": len(clients[0].experts) if clients else 0,
                    "rounds": num_rounds
                }
                
            elif "Stage 5" in stage:
                status_text.text("⏳ Running Stage 5...")
                progress_bar.progress(30)
                server, clients, meta_learner = run_meta_learning(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                result_data = {
                    "clients": len(clients),
                    "rounds": num_rounds,
                    "meta_updates": meta_learner.meta_updates
                }
                
            elif "Stage 6" in stage:
                status_text.text("⏳ Running Stage 6...")
                progress_bar.progress(30)
                server, clients, meta_learner = run_p2p_gossip(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    input_dim=input_dim,
                    output_dim=output_dim
                )
                result_data = {
                    "clients": len(clients),
                    "experts_per_client": len(clients[0].experts) if clients else 0,
                    "rounds": num_rounds
                }
                
            elif "Stage 8" in stage:
                status_text.text("⏳ Running Stage 8...")
                progress_bar.progress(30)
                clients, meta_learner, insights = run_structure_discovery(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    input_dim=input_dim
                )
                
                # Extract metrics from insights
                recon_losses = []
                replay_errors = []
                for insight in insights:
                    if "expert_results" in insight:
                        results = insight["expert_results"]
                        if "StructureExpert" in results:
                            metrics = results["StructureExpert"]["metrics"]
                            if "avg_reconstruction_loss" in metrics:
                                recon_losses.append(metrics["avg_reconstruction_loss"])
                        if "MemoryConsolidationExpert" in results:
                            metrics = results["MemoryConsolidationExpert"]["metrics"]
                            if "avg_replay_error" in metrics:
                                replay_errors.append(metrics["avg_replay_error"])
                
                result_data = {
                    "clients": len(clients),
                    "rounds": num_rounds,
                    "total_insights": len(insights),
                    "avg_reconstruction_loss": np.mean(recon_losses) if recon_losses else 0,
                    "avg_replay_error": np.mean(replay_errors) if replay_errors else 0,
                    "recon_losses": recon_losses,
                    "replay_errors": replay_errors
                }
            
            progress_bar.progress(90)
            status_text.text("✅ Experiment Complete!")
            progress_bar.progress(100)
            
            # Display logs
            st.markdown('<div class="log-box">', unsafe_allow_html=True)
            log_contents = log_capture.getvalue()
            # Show last 50 lines
            log_lines = log_contents.split('\n')
            st.code('\n'.join(log_lines[-50:]), language='log')
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            status_text.text(f"❌ Error: {str(e)}")
            st.error(f"Execution failed: {str(e)}")
            result_data = {}
        
        finally:
            # Remove handler
            logger.removeHandler(handler)
    
    # Display metrics
    with metrics_container:
        if result_data:
            st.success("✅ Experiment Complete!")
            
            # Display key metrics
            if "clients" in result_data:
                st.metric("Clients", result_data["clients"])
            if "rounds" in result_data:
                st.metric("Rounds", result_data["rounds"])
            if "insights" in result_data:
                st.metric("Insights Collected", result_data["insights"])
            if "experts_per_client" in result_data:
                st.metric("Experts per Client", result_data["experts_per_client"])
            if "meta_updates" in result_data:
                st.metric("Meta-Updates", result_data["meta_updates"])
            if "total_insights" in result_data:
                st.metric("Total Insights", result_data["total_insights"])
            
            # Stage 8 specific metrics and charts
            if "Stage 8" in stage and "recon_losses" in result_data:
                st.markdown("---")
                st.subheader("📉 Structure Discovery Metrics")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Avg Reconstruction Loss",
                        f"{result_data['avg_reconstruction_loss']:.4f}"
                    )
                with col_b:
                    st.metric(
                        "Avg Replay Error",
                        f"{result_data['avg_replay_error']:.4f}"
                    )
                
                # Plot reconstruction loss over insights
                if result_data["recon_losses"]:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(result_data["recon_losses"], marker='o', linestyle='-', color='#1f77b4')
                    ax.set_xlabel("Insight Index")
                    ax.set_ylabel("Reconstruction Loss")
                    ax.set_title("Reconstruction Loss Evolution")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
else:
    with output_container:
        st.info("👈 Configure parameters in the sidebar and click **Run Experiment** to start!")
        
        # Show example visualization
        st.subheader("Example: Structure Discovery Evolution")
        fig, ax = plt.subplots(figsize=(10, 4))
        rounds = np.arange(1, 6)
        recon_loss = [2.6144, 1.8348, 1.3729, 1.0368, 0.8211]
        replay_error = [0.1169, 0.0917, 0.0262, 0.0696, 0.0378]
        
        ax.plot(rounds, recon_loss, marker='o', label='Reconstruction Loss', linewidth=2)
        ax.plot(rounds, replay_error, marker='s', label='Replay Error', linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Value")
        ax.set_title("Stage 8: Self-Supervised Learning Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with metrics_container:
        st.info("No experiment run yet. Metrics will appear here after execution.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Scarcity Framework (SF-HFE)</b></p>
    <p>Hierarchical Federated Ensemble with Expert Routing</p>
    <p>Stages 1-8 Implemented ✓</p>
</div>
""", unsafe_allow_html=True)

