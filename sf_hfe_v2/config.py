"""
SF-HFE Configuration
Online Continual Learning System with Zero Initial Data
"""

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

SYSTEM_CONFIG = {
    "version": "1.0.0",
    "mode": "online_continual",
    "developer_has_data": False,  # Critical: Developer starts with ZERO data
}

# ============================================================================
# DATA STREAMING CONFIGURATION
# ============================================================================

STREAM_CONFIG = {
    "mini_batch_size": 32,          # Samples per mini-batch
    "update_frequency": "per_batch", # Update after every mini-batch
    "buffer_timeout": 5.0,           # Max seconds to wait for batch (if stream slow)
    "shuffle_buffer": 128,           # Pre-shuffle buffer size
}

# ============================================================================
# EXPERT SYSTEM CONFIGURATION (10 Experts + 1 Router)
# ============================================================================

EXPERT_CONFIG = {
    "num_experts": 10,
    "router_type": "contextual_bandit",  # UCB-based routing
    "expert_hidden_dims": [64, 32],      # Hidden layer sizes
    "activation": "relu",
    "dropout": 0.1,
    
    # Expert specialization
    "experts": {
        # Core Structure Experts (3)
        0: {"name": "GeometryExpert", "type": "structure", "focus": "PCA_manifold"},
        1: {"name": "TemporalExpert", "type": "structure", "focus": "LSTM_sequential"},
        2: {"name": "ReconstructionExpert", "type": "structure", "focus": "VAE_fidelity"},
        
        # Intelligence Experts (2)
        3: {"name": "CausalInferenceExpert", "type": "intelligence", "focus": "DAG_discovery"},
        4: {"name": "DriftDetectionExpert", "type": "intelligence", "focus": "KL_divergence"},
        
        # Guardrail Experts (2)
        5: {"name": "GovernanceExpert", "type": "guardrail", "focus": "constraint_validation"},
        6: {"name": "StatisticalConsistencyExpert", "type": "guardrail", "focus": "outlier_detection"},
        
        # Specialized Experts (3)
        7: {"name": "PeerSelectionExpert", "type": "relational", "focus": "cosine_similarity"},
        8: {"name": "MetaAdaptationExpert", "type": "control", "focus": "lr_scheduling"},
        9: {"name": "MemoryConsolidationExpert", "type": "memory", "focus": "latent_replay"},
    },
}

# ============================================================================
# ROUTER CONFIGURATION (Cross-Dimension Expert)
# ============================================================================

ROUTER_CONFIG = {
    "type": "contextual_bandit",
    "algorithm": "UCB",              # Upper Confidence Bound
    "top_k_active": 3,               # Activate top-3 experts per batch
    "exploration_bonus": 0.1,        # Exploration vs exploitation
    "ema_alpha": 0.95,               # EMA smoothing for stability
    "entropy_regularization": 0.01,  # Prevent router collapse
    "update_frequency": "per_batch",
    "context_dims": 32,              # Learned context representation
}

# ============================================================================
# MEMORY CONFIGURATION (3-Tier Hierarchical)
# ============================================================================

MEMORY_CONFIG = {
    # Tier 1: Recent Buffer (raw samples)
    "recent_buffer_size": 500,
    "recent_buffer_type": "fifo",
    
    # Tier 2: Compressed Memory (latent vectors)
    "compressed_size": 2000,
    "compressed_type": "reservoir",
    "compression_dim": 16,           # VAE latent dimension
    
    # Tier 3: Critical Anchors (boundary cases)
    "critical_anchors_size": 100,
    "anchor_selection": "uncertainty",  # High uncertainty samples
    
    # Replay settings
    "replay_batch_size": 32,
    "replay_frequency": 10,          # Every 10 mini-batches
    "replay_mix": {                  # Sampling from each tier
        "recent": 16,
        "compressed": 8,
        "critical": 8,
    },
}

# ============================================================================
# META-LEARNING CONFIGURATION (MAML-based)
# ============================================================================

META_LEARNING_CONFIG = {
    "algorithm": "online_maml",
    
    # Trigger conditions (OR logic - any triggers meta-learning)
    "triggers": {
        "sample_count": 1000,        # Every 1000 samples
        "time_seconds": 300,         # Every 5 minutes
        "drift_threshold": 0.05,     # KL divergence > 0.05
        "performance_drop": 0.15,    # Loss increase > 15%
    },
    
    # MAML parameters
    "inner_steps": 5,                # Inner loop adaptation steps
    "inner_lr": 0.01,                # Inner loop learning rate
    "outer_lr": 0.001,               # Outer loop (meta) learning rate
    "second_order": False,           # First-order MAML (faster)
    
    # Per-expert learning rates (adaptive)
    "expert_lr_init": 0.001,
    "expert_lr_min": 0.0001,
    "expert_lr_max": 0.01,
}

# ============================================================================
# FEDERATED LEARNING CONFIGURATION (Insight-Based)
# ============================================================================

FL_CONFIG = {
    "communication_rounds": 10000,   # Essentially continuous
    "clients_per_round": "all",      # All available clients
    "insight_frequency": 50,         # Generate insights every N batches
    
    # Insight generation (NOT raw data/weights)
    "insight_types": [
        "expert_activation_frequency",
        "loss_statistics",
        "gradient_norms",
        "router_decisions",
        "memory_utilization",
        "drift_indicators",
    ],
    
    # Aggregation
    "aggregation_method": "weighted_mean",  # Weighted by statistical consistency
    "min_clients": 1,                       # Can work with single client (dev as user)
    
    # Privacy
    "differential_privacy": False,   # Can add later
    "secure_aggregation": False,     # Can add later
}

# ============================================================================
# P2P GOSSIP CONFIGURATION
# ============================================================================

P2P_CONFIG = {
    "enabled": True,
    "topology": "adaptive",          # Peer-Selection Expert builds topology
    
    # Peer selection
    "top_k_peers": 3,                # Connect to 3 most similar peers
    "similarity_metric": "cosine",
    "similarity_ema_alpha": 0.9,     # Smooth peer selection
    "hysteresis_threshold": 0.05,    # Prevent peer churn
    
    # Gossip exchange
    "exchange_frequency": 180,       # Every 3 minutes
    "exchange_type": "active_experts_only",  # Only most-used expert weights
    "exchange_top_n": 3,             # Exchange weights of top-3 active experts
    
    # Local aggregation
    "aggregation_weight": 0.5,       # 50% self, 50% peer average
}

# ============================================================================
# LEARNING DYNAMICS
# ============================================================================

LEARNING_CONFIG = {
    # Base learning rate
    "base_lr": 0.001,
    "lr_scheduler": "adaptive",      # Per-expert adaptive LR
    "lr_decay_factor": 0.95,         # Decay per 1000 samples
    "lr_warmup_steps": 100,          # Warmup for first 100 batches
    
    # Optimization
    "optimizer": "adam",
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "momentum": 0.9,
    
    # Stability
    "loss_smoothing_ema": 0.95,
    "gradient_noise_scale": 0.01,   # Optional noise for exploration
}

# ============================================================================
# CONTINUAL LEARNING SAFEGUARDS
# ============================================================================

STABILITY_CONFIG = {
    # Anti-forgetting
    "elastic_weight_consolidation": True,  # EWC for important weights
    "ewc_lambda": 0.4,                     # EWC regularization strength
    
    # Router-Expert oscillation prevention
    "router_momentum": 0.9,
    "expert_activation_ema": 0.95,
    
    # Meta-adaptation safeguards
    "lr_change_max": 0.5,            # Max LR change per update (50%)
    "freeze_threshold": 0.001,       # Freeze expert if gradient < threshold
    
    # Peer graph stability
    "peer_update_cooldown": 60,      # Min 60s between peer changes
    "min_interaction_before_trust": 5,  # Need 5 exchanges before full trust
}

# ============================================================================
# MONITORING & LOGGING
# ============================================================================

MONITORING_CONFIG = {
    "log_level": "INFO",
    "log_frequency": 100,            # Log every 100 batches
    
    # Metrics to track
    "metrics": [
        "loss_per_expert",
        "router_entropy",
        "expert_activation_counts",
        "memory_utilization",
        "drift_score",
        "meta_learning_frequency",
        "p2p_exchange_count",
    ],
    
    # Checkpointing
    "checkpoint_frequency": 1000,    # Every 1000 samples
    "checkpoint_dir": "./checkpoints",
    "keep_n_checkpoints": 5,
}

# ============================================================================
# DATA CONFIGURATION (For Testing)
# ============================================================================

DATA_CONFIG = {
    # Synthetic stream for testing
    "synthetic_stream": {
        "enabled": True,
        "num_features": 20,
        "num_classes": 5,
        "stream_length": 10000,
        
        # Concept drift simulation
        "concept_drift": {
            "enabled": True,
            "drift_points": [2500, 5000, 7500],  # Distribution changes
            "drift_magnitude": 0.3,
        },
        
        # Noise
        "noise_level": 0.1,
    },
    
    # Real data (can be enabled later)
    "real_data": {
        "enabled": False,
        "type": None,  # Will be set by user
        "path": None,
    },
}

