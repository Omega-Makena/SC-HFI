"""
SF-HFE Configuration
Online Continual Learning System with Zero Initial Data
"""

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

SYSTEM_CONFIG = {
"version": "2.0.0",
"mode": "online_continual",
"developer_has_data": False, # Critical: Developer starts with ZERO data

# REPRODUCIBILITY (P0 - Critical for research)
"random_seed": 42,
"deterministic": True, # Enable deterministic mode

# Logging configuration
"log_level": "INFO",
"log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
"log_file": "logs/federated_learning.log",
}

# ============================================================================
# DATA STREAMING CONFIGURATION
# ============================================================================

STREAM_CONFIG = {
"mini_batch_size": 32, # Samples per mini-batch
"update_frequency": "per_batch", # Update after every mini-batch
"buffer_timeout": 5.0, # Max seconds to wait for batch (if stream slow)
"shuffle_buffer": 128, # Pre-shuffle buffer size
}

# ============================================================================
# EXPERT SYSTEM CONFIGURATION (30 Experts + Advanced Router)
# ============================================================================

EXPERT_CONFIG = {
"num_experts": 30,
"router_type": "advanced_online_learning", # Advanced online learning router
"expert_hidden_dims": [128, 64, 32], # Hidden layer sizes
"activation": "relu",
"dropout": 0.1,

# Online learning parameters
"online_learning": {
"learning_rate": 0.001,
"adaptation_rate": 0.01,
"exploration_rate": 0.1,
"exploration_decay": 0.99,
"performance_window": 50,
"context_memory_size": 1000,
"meta_learning": {
"learning_rate": 0.01,
"memory_size": 1000
},
"performance_predictor": {
"prediction_window": 20
}
},

# Expert specialization by relation types
"experts": {
# Structural Experts (1-4)
1: {"name": "SchemaMapperExpert", "relation_types": ["structural"], "focus": "schema_detection"},
2: {"name": "TypeFormatExpert", "relation_types": ["structural"], "focus": "type_detection"},
3: {"name": "MissingnessNoiseExpert", "relation_types": ["structural", "statistical"], "focus": "quality_assessment"},
4: {"name": "ScalingEncodingExpert", "relation_types": ["structural"], "focus": "preprocessing"},

# Statistical Experts (5-8)
5: {"name": "DescriptiveExpert", "relation_types": ["statistical"], "focus": "descriptive_analysis"},
6: {"name": "CorrelationExpert", "relation_types": ["statistical"], "focus": "correlation_analysis"},
7: {"name": "DensityExpert", "relation_types": ["statistical"], "focus": "density_estimation"},
8: {"name": "AnomalyExpert", "relation_types": ["statistical"], "focus": "anomaly_detection"},

# Temporal Experts (9-12)
9: {"name": "TrendExpert", "relation_types": ["temporal"], "focus": "trend_detection"},
10: {"name": "DriftExpert", "relation_types": ["temporal", "statistical"], "focus": "drift_detection"},
11: {"name": "CyclicExpert", "relation_types": ["temporal"], "focus": "seasonality_detection"},
12: {"name": "TemporalCausalityExpert", "relation_types": ["temporal", "causal"], "focus": "temporal_causality"},

# Relational/Interactional Experts (13-16)
13: {"name": "GraphBuilderExpert", "relation_types": ["relational", "interactional"], "focus": "graph_construction"},
14: {"name": "InfluenceExpert", "relation_types": ["relational", "causal"], "focus": "influence_detection"},
15: {"name": "GroupDynamicsExpert", "relation_types": ["relational", "statistical"], "focus": "group_detection"},
16: {"name": "FeedbackLoopExpert", "relation_types": ["relational", "causal"], "focus": "feedback_detection"},

# Causal Experts (17-20)
17: {"name": "CausalDiscoveryExpert", "relation_types": ["causal"], "focus": "causal_discovery"},
18: {"name": "CounterfactualExpert", "relation_types": ["causal"], "focus": "counterfactual_analysis"},
19: {"name": "MediationExpert", "relation_types": ["causal"], "focus": "mediation_analysis"},
20: {"name": "PolicyEffectExpert", "relation_types": ["causal"], "focus": "policy_simulation"},

# Semantic/Contextual Experts (21-24)
21: {"name": "ContextualExpert", "relation_types": ["semantic", "contextual"], "focus": "context_analysis"},
22: {"name": "DomainOntologyExpert", "relation_types": ["semantic"], "focus": "ontology_learning"},
23: {"name": "CrossDomainTransferExpert", "relation_types": ["semantic"], "focus": "domain_transfer"},
24: {"name": "RepresentationConsistencyExpert", "relation_types": ["semantic"], "focus": "consistency_checking"},

# Integrative/Cognitive Experts (25-30)
25: {"name": "CognitiveExpert", "relation_types": ["cognitive", "integrative"], "focus": "cognitive_integration"},
26: {"name": "SimulationExpert", "relation_types": ["cognitive", "predictive"], "focus": "simulation"},
27: {"name": "ForecastExpert", "relation_types": ["cognitive", "predictive"], "focus": "forecasting"},
28: {"name": "MetaFeedbackExpert", "relation_types": ["cognitive", "meta"], "focus": "meta_feedback"},
29: {"name": "MemoryCuratorExpert", "relation_types": ["cognitive", "memory"], "focus": "memory_curation"},
30: {"name": "EthicalConstraintExpert", "relation_types": ["cognitive", "ethical"], "focus": "ethical_validation"},
},

# Expert group configurations
"expert_groups": {
"structural": {"experts": [1, 2, 3, 4], "priority": 0.8},
"statistical": {"experts": [5, 6, 7, 8], "priority": 0.7},
"temporal": {"experts": [9, 10, 11, 12], "priority": 0.7},
"relational": {"experts": [13, 14, 15, 16], "priority": 0.6},
"causal": {"experts": [17, 18, 19, 20], "priority": 0.6},
"semantic": {"experts": [21, 22, 23, 24], "priority": 0.5},
"cognitive": {"experts": [25, 26, 27, 28, 29, 30], "priority": 0.9}
}
}

# ============================================================================
# ROUTER CONFIGURATION (Cross-Dimension Expert)
# ============================================================================

ROUTER_CONFIG = {
"type": "contextual_bandit",
"algorithm": "UCB", # Upper Confidence Bound
"top_k_active": 3, # Activate top-3 experts per batch
"exploration_bonus": 0.1, # Exploration vs exploitation
"ema_alpha": 0.95, # EMA smoothing for stability
"entropy_regularization": 0.01, # Prevent router collapse
"update_frequency": "per_batch",
"context_dims": 32, # Learned context representation
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
"compression_dim": 16, # VAE latent dimension

# Tier 3: Critical Anchors (boundary cases)
"critical_anchors_size": 100,
"anchor_selection": "uncertainty", # High uncertainty samples

# Replay settings
"replay_batch_size": 32,
"replay_frequency": 10, # Every 10 mini-batches
"replay_mix": { # Sampling from each tier
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
"sample_count": 1000, # Every 1000 samples
"time_seconds": 300, # Every 5 minutes
"drift_threshold": 0.05, # KL divergence > 0.05
"performance_drop": 0.15, # Loss increase > 15%
},

# MAML parameters
"inner_steps": 5, # Inner loop adaptation steps
"inner_lr": 0.01, # Inner loop learning rate
"outer_lr": 0.001, # Outer loop (meta) learning rate
"second_order": False, # First-order MAML (faster)

# Per-expert learning rates (adaptive)
"expert_lr_init": 0.001,
"expert_lr_min": 0.0001,
"expert_lr_max": 0.01,
}

# ============================================================================
# FEDERATED LEARNING CONFIGURATION (Insight-Based)
# ============================================================================

FL_CONFIG = {
"communication_rounds": 10000, # Essentially continuous
"clients_per_round": "all", # All available clients
"insight_frequency": 50, # Generate insights every N batches
"max_insights": 10000, # Bounded memory storage

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
"aggregation_method": "weighted_mean", # Weighted by statistical consistency
"min_clients": 1, # Can work with single client (dev as user)

# Privacy
"differential_privacy": False, # Can add later
"secure_aggregation": False, # Can add later

# Rate limiting
"max_requests_per_minute": 100,
"request_timeout": 30,
}

# ============================================================================
# P2P GOSSIP CONFIGURATION
# ============================================================================

P2P_CONFIG = {
"enabled": True,
"topology": "adaptive", # Peer-Selection Expert builds topology

# Peer selection
"top_k_peers": 3, # Connect to 3 most similar peers
"similarity_metric": "cosine",
"similarity_ema_alpha": 0.9, # Smooth peer selection
"hysteresis_threshold": 0.05, # Prevent peer churn

# Gossip exchange
"exchange_frequency": 5, # Every 5 seconds (for testing)
"exchange_type": "active_experts_only", # Only most-used expert weights
"exchange_top_n": 3, # Exchange weights of top-3 active experts

# Local aggregation
"aggregation_weight": 0.5, # 50% self, 50% peer average
}

# ============================================================================
# LEARNING DYNAMICS
# ============================================================================

LEARNING_CONFIG = {
# Base learning rate
"base_lr": 0.001,
"lr_scheduler": "adaptive", # Per-expert adaptive LR
"lr_decay_factor": 0.95, # Decay per 1000 samples
"lr_warmup_steps": 100, # Warmup for first 100 batches

# Optimization
"optimizer": "adam",
"weight_decay": 1e-5,
"gradient_clip": 1.0,
"momentum": 0.9,

# Stability
"loss_smoothing_ema": 0.95,
"gradient_noise_scale": 0.01, # Optional noise for exploration
}

# ============================================================================
# CONTINUAL LEARNING SAFEGUARDS
# ============================================================================

STABILITY_CONFIG = {
# Anti-forgetting
"elastic_weight_consolidation": True, # EWC for important weights
"ewc_lambda": 0.4, # EWC regularization strength

# Router-Expert oscillation prevention
"router_momentum": 0.9,
"expert_activation_ema": 0.95,

# Meta-adaptation safeguards
"lr_change_max": 0.5, # Max LR change per update (50%)
"freeze_threshold": 0.001, # Freeze expert if gradient < threshold

# Peer graph stability
"peer_update_cooldown": 60, # Min 60s between peer changes
"min_interaction_before_trust": 5, # Need 5 exchanges before full trust
}

# ============================================================================
# MONITORING & LOGGING
# ============================================================================

MONITORING_CONFIG = {
"log_level": "INFO",
"log_frequency": 100, # Log every 100 batches

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
"checkpoint_frequency": 1000, # Every 1000 samples
"checkpoint_dir": "./checkpoints",
"keep_n_checkpoints": 5,
}

# ============================================================================
# EVALUATION CONFIGURATION (P1 - Essential for Research)
# ============================================================================

EVALUATION_CONFIG = {
# Test split
"test_split": 0.2, # 20% for testing
"eval_frequency": 50, # Evaluate every N batches

# Fairness metrics (P1 - Critical)
"compute_fairness": True,
"fairness_metrics": [
"hfi", # Healthcare Fairness Index
"worst_case_ratio", # Max/Avg performance
"coefficient_variation", # Relative dispersion
"min_max_range", # Performance range
],

# Per-client evaluation
"per_client_metrics": True,
"track_client_variance": True,

# Baseline comparison
"compare_to_fedavg": True, # Run FedAvg baseline
"baseline_rounds": 10, # FedAvg rounds for comparison
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
"drift_points": [2500, 5000, 7500], # Distribution changes
"drift_magnitude": 0.3,
},

# Noise
"noise_level": 0.1,
},

# Real data (can be enabled later)
"real_data": {
"enabled": False,
"type": None, # Will be set by user
"path": None,
},
}

