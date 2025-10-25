"""
Display OMEO Insights in Terminal
"""

import requests
import json
from datetime import datetime

def display_omoe_insights():
"""Display OMEO insights in a formatted way"""
try:
# Get insights from API
response = requests.get('http://127.0.0.1:5000/api/insights')
if response.status_code != 200:
print("ERROR: No insights available")
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
return

insights = response.json()

print("=" * 80)
print("NEURAL OMEO SYSTEM INSIGHTS")
print("=" * 80)
print(f"Data Samples: {insights.get('data_samples', 'N/A')}")
print(f"Domain: {insights.get('domain', 'N/A')}")
print(f"Users Created: {insights.get('users_created', 'N/A')}")
print(f"Processing Time: {insights.get('processing_time', 'N/A')}")
print()

# Display OMEO Tier Insights
if 'omoe_insights' in insights:
print("OMEO NEURAL TIER ANALYSIS")
print("-" * 50)

tier_names = {
'tier1_structural': 'Tier 1: Structural Understanding',
'tier2_relational': 'Tier 2: Relational Understanding',
'tier3_dynamical': 'Tier 3: Dynamical Understanding',
'tier4_semantic': 'Tier 4: Semantic Understanding',
'tier5_projective': 'Tier 5: Projective Understanding',
'tier6_meta': 'Tier 6: Meta Understanding'
}

for tier_key, tier_name in tier_names.items():
if tier_key in insights['omoe_insights']:
tier_data = insights['omoe_insights'][tier_key]
confidence = tier_data.get('confidence', 0)
adaptation_count = tier_data.get('adaptation_count', 0)

print(f"\n{tier_name}")
print(f" Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
print(f" Adaptations: {adaptation_count}")

# Show key insights
key_insights = []
for key, value in tier_data.items():
if key not in ['confidence', 'adaptation_count']:
if isinstance(value, (list, tuple)):
key_insights.append(f"{key}: {len(value)} items")
elif isinstance(value, dict):
key_insights.append(f"{key}: {len(value)} properties")
else:
key_insights.append(f"{key}: {str(value)[:50]}")

if key_insights:
print(f" Key Insights:")
for insight in key_insights[:3]: # Show first 3 insights
print(f" - {insight}")

# Display Performance Metrics
if 'omoe_performance' in insights:
print("\nPERFORMANCE METRICS")
print("-" * 30)
for tier_key, perf_data in insights['omoe_performance'].items():
tier_name = tier_names.get(tier_key, tier_key)
performance = perf_data.get('performance', 0)
updates = perf_data.get('updates', 0)
print(f" {tier_name}: {performance:.1f}% (Updates: {updates})")

# Display Simulations
if 'simulations' in insights:
print("\nSIMULATIONS GENERATED")
print("-" * 30)
for sim_type, sim_data in insights['simulations'].items():
print(f" {sim_type.title()}: {sim_data.get('scenario', 'N/A')}")
if 'confidence' in sim_data:
print(f" Confidence: {sim_data['confidence']:.3f}")

# Display System Metrics
if 'system_metrics' in insights:
metrics = insights['system_metrics']
print("\nSYSTEM METRICS")
print("-" * 20)
print(f" Active Users: {metrics.get('active_users', 0)}")
print(f" OMEO Models: {metrics.get('omoe_models', 0)}")
print(f" Model Updates: {metrics.get('model_updates', 0)}")
print(f" Gossip Exchanges: {metrics.get('gossip_exchanges', 0)}")
print(f" Stored Insights: {metrics.get('stored_insights', 0)}")
print(f" Avg Performance: {metrics.get('avg_performance', 0):.1f}%")

print("\n" + "=" * 80)
print("SUCCESS: OMEO System is processing data successfully!")
print("Dashboard: http://127.0.0.1:5000")
print("=" * 80)

except requests.exceptions.ConnectionError:
print("ERROR: Cannot connect to OMEO system")
print("Make sure the backend service is running on http://127.0.0.1:5000")
except Exception as e:
print(f"ERROR: {e}")

if __name__ == "__main__":
display_omoe_insights()
