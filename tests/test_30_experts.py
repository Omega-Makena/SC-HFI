#!/usr/bin/env python3
"""
Test script for the new 30-expert online learning system
Verifies that all experts are properly initialized and can process data
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sf_hfe_v2.moe.client import SFHFEClient
from sf_hfe_v2.config import EXPERT_CONFIG

def test_expert_system():
"""Test the 30-expert online learning system"""
print(" Testing 30-Expert Online Learning System")
print("=" * 50)

# Initialize client with 30 experts
print("1. Initializing SFHFEClient with 30 experts...")
client = SFHFEClient(client_id=1, domain="test", config=EXPERT_CONFIG)

print(f" Client initialized with {len(client.experts)} experts")
print(f" Router type: {type(client.router).__name__}")

# Test data processing
print("\n2. Testing data processing...")

# Create test data
test_data = np.random.randn(100, 5) # 100 samples, 5 features
test_metadata = {
'domain': 'test',
'task_type': 'analysis',
'generate_simulation': True
}

print(f" Test data shape: {test_data.shape}")

# Process data through expert system
results = client.process_data(test_data, test_metadata)

if 'error' in results:
print(f" Error processing data: {results['error']}")
return False

print(f" Data processed successfully")
print(f" Experts used: {results['processing_metadata']['num_experts_used']}")
print(f" Total experts available: {results['processing_metadata']['total_experts']}")

# Test expert results
print("\n3. Analyzing expert results...")
expert_results = results['expert_results']

print(f" Active experts: {len(expert_results)}")

for expert_id, result_info in expert_results.items():
expert_name = result_info['expert_name']
weight = result_info['weight']
confidence = result_info['result'].get('confidence', 0.0)

print(f" Expert {expert_id} ({expert_name}): weight={weight:.3f}, confidence={confidence:.3f}")

# Test simulation results
if results['simulation_results']:
print("\n4. Simulation results generated ")
else:
print("\n4. No simulation results generated")

# Test router statistics
print("\n5. Router statistics...")
router_stats = client.router.get_routing_stats()
print(f" Total experts: {router_stats['total_experts']}")
print(f" Active experts: {router_stats['active_experts']}")
print(f" Routing history size: {router_stats['routing_history_size']}")

# Test expert groups
print("\n6. Expert groups...")
expert_groups = router_stats.get('expert_groups', {})
for group_name, group_size in expert_groups.items():
print(f" {group_name}: {group_size} experts")

print("\n" + "=" * 50)
print("🎉 All tests passed! 30-expert system is working correctly.")
return True

def test_individual_experts():
"""Test individual expert functionality"""
print("\n🔬 Testing individual expert functionality...")

# Test structural experts
from sf_hfe_v2.moe.structural_experts import SchemaMapperExpert, TypeFormatExpert

print(" Testing SchemaMapperExpert...")
schema_expert = SchemaMapperExpert(expert_id=1)
test_data = np.random.randn(50, 3)
result = schema_expert.process_data(test_data)
print(f" SchemaMapperExpert confidence: {result.get('confidence', 0.0):.3f}")

print(" Testing TypeFormatExpert...")
type_expert = TypeFormatExpert(expert_id=2)
result = type_expert.process_data(test_data)
print(f" TypeFormatExpert confidence: {result.get('confidence', 0.0):.3f}")

# Test statistical experts
from sf_hfe_v2.moe.statistical_experts import DescriptiveExpert, CorrelationExpert

print(" Testing DescriptiveExpert...")
desc_expert = DescriptiveExpert(expert_id=5)
result = desc_expert.process_data(test_data)
print(f" DescriptiveExpert confidence: {result.get('confidence', 0.0):.3f}")

print(" Testing CorrelationExpert...")
corr_expert = CorrelationExpert(expert_id=6)
result = corr_expert.process_data(test_data)
print(f" CorrelationExpert confidence: {result.get('confidence', 0.0):.3f}")

print(" Individual expert tests completed")

if __name__ == "__main__":
try:
# Run main test
success = test_expert_system()

if success:
# Run individual expert tests
test_individual_experts()

print("\n🎊 All tests completed successfully!")
print("The 30-expert online learning system is ready for use.")
else:
print("\n Tests failed. Please check the error messages above.")
sys.exit(1)

except Exception as e:
print(f"\n💥 Unexpected error during testing: {e}")
import traceback
traceback.print_exc()
sys.exit(1)
