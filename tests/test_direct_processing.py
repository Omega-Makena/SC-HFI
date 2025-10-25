"""
Direct test of OMEO processing function
"""

import pandas as pd
import sys
import os

# Add the sf_hfe_v2 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sf_hfe_v2'))

def test_direct_processing():
"""Test OMEO processing directly"""
print("Testing OMEO processing directly...")

# Load market data
df = pd.read_csv('Market_Prices.csv')
print(f"Loaded market data: {df.shape}")

# Import the processing function
from backend_service import process_data_with_experts

# Test processing
try:
print("Calling process_data_with_experts...")
result = process_data_with_experts(df, "economic")
print(f"Processing result: {result is not None}")
if result:
print(f"Result keys: {list(result.keys())}")
print(f"OMEO insights: {list(result.get('omoe_insights', {}).keys())}")
else:
print("Result is None - processing failed")
except Exception as e:
print(f"Error in processing: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
test_direct_processing()
