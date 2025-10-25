"""
Test individual neural tiers with market data to identify issues
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the sf_hfe_v2 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sf_hfe_v2.moe.neural_tiers import (
Tier1Structural, Tier2Relational, Tier3Dynamical,
Tier4Semantic, Tier5Projective, Tier6Meta
)
from sf_hfe_v2.moe.gate import DomainRouter

def test_market_data_processing():
"""Test market data processing through individual tiers"""
print("Testing Market Data Processing...")

# Load market data
try:
df = pd.read_csv('Market_Prices.csv')
print(f"Loaded market data: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types: {df.dtypes.to_dict()}")
print(f"Missing values: {df.isnull().sum().to_dict()}")
print(f"Sample data:")
print(df.head())
print()
except Exception as e:
print(f"Error loading market data: {e}")
return

# Test Domain Router
try:
print("=== Testing Domain Router ===")
gate = DomainRouter(config={"domain": "economic"})
routing_result = gate.route_data(df, {"domain": "economic"})
print(f"Routing result: {routing_result}")
print()
except Exception as e:
print(f"Domain Router error: {e}")
return

# Test Tier 1
try:
print("=== Testing Tier 1 (Structural) ===")
tier1 = Tier1Structural(config={"domain": "economic"})
tier1_result = tier1.analyze(df, routing_result)
print(f"Tier 1 result: {tier1_result}")
print()
except Exception as e:
print(f"Tier 1 error: {e}")
import traceback
traceback.print_exc()
return

# Test Tier 2
try:
print("=== Testing Tier 2 (Relational) ===")
tier2 = Tier2Relational(config={"domain": "economic"})
tier2_result = tier2.analyze(df, tier1_result)
print(f"Tier 2 result: {tier2_result}")
print()
except Exception as e:
print(f"Tier 2 error: {e}")
import traceback
traceback.print_exc()
return

# Test Tier 3
try:
print("=== Testing Tier 3 (Dynamical) ===")
tier3 = Tier3Dynamical(config={"domain": "economic"})
tier3_result = tier3.analyze(df, tier2_result)
print(f"Tier 3 result: {tier3_result}")
print()
except Exception as e:
print(f"Tier 3 error: {e}")
import traceback
traceback.print_exc()
return

# Test Tier 4
try:
print("=== Testing Tier 4 (Semantic) ===")
tier4 = Tier4Semantic(config={"domain": "economic"})
tier4_result = tier4.analyze(df, tier3_result)
print(f"Tier 4 result: {tier4_result}")
print()
except Exception as e:
print(f"Tier 4 error: {e}")
import traceback
traceback.print_exc()
return

# Test Tier 5
try:
print("=== Testing Tier 5 (Projective) ===")
tier5 = Tier5Projective(config={"domain": "economic"})
tier5_result = tier5.analyze(df, tier4_result)
print(f"Tier 5 result: {tier5_result}")
print()
except Exception as e:
print(f"Tier 5 error: {e}")
import traceback
traceback.print_exc()
return

# Test Tier 6
try:
print("=== Testing Tier 6 (Meta Understanding) ===")
tier6 = Tier6Meta(config={"domain": "economic"})
tier6_result = tier6.analyze(df, tier5_result)
print(f"Tier 6 result: {tier6_result}")
print()
except Exception as e:
print(f"Tier 6 error: {e}")
import traceback
traceback.print_exc()
return

print("=== All Tiers Completed Successfully ===")

if __name__ == "__main__":
test_market_data_processing()
