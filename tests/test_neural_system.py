"""
Test script to identify data processing issues in the neural OMEO system
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_neural_models():
"""Test each neural model individually"""
print("Testing Neural OMEO System...")

# Create test data
test_data = pd.DataFrame({
'feature1': np.random.randn(100),
'feature2': np.random.randn(100),
'feature3': np.random.randn(100),
'target': np.random.randn(100)
})

print(f"Created test data: {test_data.shape}")

try:
# Test Tier 1 (Structural)
print("\n=== Testing Tier 1 (Structural) ===")
from moe.neural_tiers import Tier1Structural
tier1 = Tier1Structural()
result1 = tier1.analyze(test_data, {})
print(f"Tier 1 Result: {result1}")

except Exception as e:
print(f"Tier 1 Error: {e}")
import traceback
traceback.print_exc()

try:
# Test Tier 2 (Relational)
print("\n=== Testing Tier 2 (Relational) ===")
from moe.neural_tiers import Tier2Relational
tier2 = Tier2Relational()
result2 = tier2.analyze(test_data, result1 if 'result1' in locals() else {})
print(f"Tier 2 Result: {result2}")

except Exception as e:
print(f"Tier 2 Error: {e}")
import traceback
traceback.print_exc()

try:
# Test Tier 3 (Dynamical)
print("\n=== Testing Tier 3 (Dynamical) ===")
from moe.neural_tiers import Tier3Dynamical
tier3 = Tier3Dynamical()
result3 = tier3.analyze(test_data, result2 if 'result2' in locals() else {})
print(f"Tier 3 Result: {result3}")

except Exception as e:
print(f"Tier 3 Error: {e}")
import traceback
traceback.print_exc()

try:
# Test Tier 4 (Semantic)
print("\n=== Testing Tier 4 (Semantic) ===")
from moe.neural_tiers import Tier4Semantic
tier4 = Tier4Semantic()
result4 = tier4.analyze(test_data, result3 if 'result3' in locals() else {})
print(f"Tier 4 Result: {result4}")

except Exception as e:
print(f"Tier 4 Error: {e}")
import traceback
traceback.print_exc()

try:
# Test Tier 5 (Projective)
print("\n=== Testing Tier 5 (Projective) ===")
from moe.neural_tiers import Tier5Projective
tier5 = Tier5Projective()
result5 = tier5.analyze(test_data, result4 if 'result4' in locals() else {})
print(f"Tier 5 Result: {result5}")

except Exception as e:
print(f"Tier 5 Error: {e}")
import traceback
traceback.print_exc()

try:
# Test Tier 6 (Meta Understanding)
print("\n=== Testing Tier 6 (Meta Understanding) ===")
from moe.neural_tiers import Tier6Meta
tier6 = Tier6Meta()
result6 = tier6.analyze(test_data, result5 if 'result5' in locals() else {})
print(f"Tier 6 Result: {result6}")

except Exception as e:
print(f"Tier 6 Error: {e}")
import traceback
traceback.print_exc()

def test_backend_processing():
"""Test the backend processing function"""
print("\n=== Testing Backend Processing ===")

try:
from backend_service import process_data_with_experts_simple
test_data = pd.DataFrame({
'feature1': np.random.randn(50),
'feature2': np.random.randn(50),
'feature3': np.random.randn(50)
})

result = process_data_with_experts_simple(test_data, "economic")
print(f"Backend Simple Processing Result: {result is not None}")
if result:
print(f"Keys: {list(result.keys())}")

except Exception as e:
print(f"Backend Processing Error: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
test_neural_models()
test_backend_processing()
