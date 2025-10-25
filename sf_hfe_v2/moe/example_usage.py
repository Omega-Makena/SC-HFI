"""
Example usage of the Unified Storage System
Shows how to use the same storage for both Federated Learning and MoE
"""

from unified_storage import UnifiedStorage
import pandas as pd
from datetime import datetime

def example_federated_learning():
"""Example of using unified storage for federated learning"""
print("=== Federated Learning Example ===")

# Initialize unified storage
storage = UnifiedStorage(config={
'max_insights': 1000,
'storage_threshold': 50
})

# Add federated learning insights
federated_insight = {
'client_id': 'client_001',
'expert_insights': {
'model_accuracy': 0.85,
'training_samples': 1000,
'loss_reduction': 0.15
},
'avg_loss': 0.25,
'total_samples': 1000,
'domain': 'economic'
}

storage.add_federated_insight(federated_insight)

# Get federated insights
federated_insights = storage.get_federated_insights()
print(f"Federated insights count: {len(federated_insights)}")

# Get insights by client
client_insights = storage.get_insights_by_client('client_001')
print(f"Client insights count: {len(client_insights)}")

# Get insights by domain
domain_insights = storage.get_insights_by_domain('economic')
print(f"Domain insights count: {len(domain_insights)}")

def example_moe_system():
"""Example of using unified storage for MoE system"""
print("\n=== MoE System Example ===")

# Initialize unified storage
storage = UnifiedStorage(config={
'max_insights': 1000,
'storage_threshold': 50
})

# Create mock MoE results
tier_results = {
'tier1': {
'fingerprint': {'quality_score': 0.8},
'schema': {'numeric_columns': 5}
},
'tier2': {
'relationship_summary': {'strong_correlations': 3}
},
'tier3': {
'dynamical_summary': {'trending_variables': 2}
},
'tier4': {
'semantic_summary': {'important_features': 4}
},
'tier5': {
'projective_summary': {'average_forecast_confidence': 0.75}
},
'tier6': {
'meta_summary': {'performance_score': 0.8}
}
}

routing_result = {
'basket_label': 'economic',
'confidence_score': 0.85,
'data_shape': (100, 10)
}

# Add MoE insights
storage.add_moe_insight(tier_results, routing_result)

# Get MoE insights
moe_insights = storage.get_moe_insights()
print(f"MoE insights count: {len(moe_insights)}")

# Get insights by tier
tier1_insights = storage.get_insights_by_tier('tier1')
print(f"Tier 1 insights count: {len(tier1_insights)}")

# Get insights by domain basket
basket_insights = storage.get_insights_by_domain_basket('economic')
print(f"Domain basket insights count: {len(basket_insights)}")

def example_combined_usage():
"""Example of using unified storage for both systems together"""
print("\n=== Combined Usage Example ===")

# Initialize unified storage
storage = UnifiedStorage(config={
'max_insights': 1000,
'storage_threshold': 50
})

# Add both types of insights
federated_insight = {
'client_id': 'client_001',
'expert_insights': {'model_accuracy': 0.85},
'avg_loss': 0.25,
'total_samples': 1000,
'domain': 'economic'
}

tier_results = {
'tier1': {'fingerprint': {'quality_score': 0.8}},
'tier2': {'relationship_summary': {'strong_correlations': 3}}
}

routing_result = {
'basket_label': 'economic',
'confidence_score': 0.85
}

storage.add_federated_insight(federated_insight)
storage.add_moe_insight(tier_results, routing_result)

# Get comprehensive statistics
stats = storage.stats()
print(f"Total insights: {stats['total_insights']}")
print(f"Unique clients: {stats['unique_clients']}")
print(f"Domains: {stats['domains']}")
print(f"Domain baskets: {stats['domain_baskets']}")
print(f"Tiers: {stats['tiers']}")

# Get aggregated metrics
metrics = storage.get_aggregated_metrics()
print(f"Aggregated metrics: {list(metrics.keys())}")

# Get meta-learning triggers
triggers = storage.get_meta_learning_triggers()
print(f"Meta-learning triggers: {len(triggers)}")

if __name__ == "__main__":
example_federated_learning()
example_moe_system()
example_combined_usage()
