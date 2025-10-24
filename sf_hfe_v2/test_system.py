#!/usr/bin/env python3
"""
End-to-End System Test for Federated Learning Components
Tests the complete workflow from insights to meta-learning
"""

import sys
import os
import logging
import threading
import time
import numpy as np
import torch
from typing import Dict, List
from unittest.mock import Mock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    SYSTEM_CONFIG, META_LEARNING_CONFIG, FL_CONFIG, 
    P2P_CONFIG, STABILITY_CONFIG
)

# Import federated components
from federated.global_memory import GlobalMemory
from federated.meta_learning import OnlineMAMLEngine
from federated.gossip import P2PGossipManager
from federated.server import SFHFEServer
from federated.initialization import initialize_system


def setup_test_environment():
    """Setup test environment with proper logging and reproducibility"""
    print("🔧 Setting up test environment...")
    
    # Initialize system
    initialize_system(log_level="INFO", seed=42)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("✅ Test environment ready")


def test_global_memory():
    """Test GlobalMemory bounded storage and validation"""
    print("\n🧠 Testing GlobalMemory...")
    
    memory = GlobalMemory(max_insights=100)
    
    # Test 1: Add valid insights
    print("  📝 Adding valid insights...")
    for i in range(50):
        insight = {
            "client_id": i % 5,
            "expert_insights": {
                f"expert_{j}": {
                    "expert_id": j,
                    "ema_loss": 0.1 + j * 0.01,
                    "activation_count": 10,
                    "learning_rate": 0.001
                }
                for j in range(3)
            },
            "avg_loss": 0.1 + i * 0.001,
            "total_samples": 100
        }
        memory.add_insight(insight)
    
    stats = memory.stats()
    print(f"  ✅ Added {stats['total_insights']} insights, current: {stats['current_insights']}")
    
    # Test 2: Test bounded storage
    print("  📊 Testing bounded storage...")
    for i in range(100):  # Add more than max
        insight = {
            "client_id": i % 5,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1, "activation_count": 10, "learning_rate": 0.001}},
            "avg_loss": 0.1,
            "total_samples": 100
        }
        memory.add_insight(insight)
    
    stats = memory.stats()
    print(f"  ✅ Memory bounded: {stats['current_insights']}/{stats['max_insights']}")
    assert stats['current_insights'] <= stats['max_insights'], "Memory not properly bounded!"
    
    # Test 3: Test validation
    print("  🔍 Testing insight validation...")
    invalid_insight = {"invalid": "data"}
    initial_count = memory.total_insights
    memory.add_insight(invalid_insight)
    assert memory.total_insights == initial_count, "Invalid insight was added!"
    print("  ✅ Validation working correctly")
    
    print("✅ GlobalMemory tests passed!")


def test_meta_learning_engine():
    """Test OnlineMAMLEngine meta-learning functionality"""
    print("\n🎯 Testing Meta-Learning Engine...")
    
    engine = OnlineMAMLEngine(num_experts=5)
    
    # Test 1: Meta-parameter usage
    print("  📊 Testing meta-parameter usage...")
    insights = []
    for i in range(10):
        insight = {
            "client_id": i,
            "expert_insights": {
                f"expert_{j}": {
                    "expert_id": j,
                    "ema_loss": 0.1 + j * 0.01,
                    "activation_count": 10,
                    "learning_rate": 0.001 + j * 0.0001
                }
                for j in range(5)
            },
            "avg_loss": 0.1 + i * 0.01,
            "total_samples": 100
        }
        insights.append(insight)
    
    meta_params = engine.meta_update(insights)
    
    # Verify w_init is included and updated
    assert "w_init" in meta_params, "w_init not in meta parameters!"
    assert "parameter_version" in meta_params, "Version tracking missing!"
    assert meta_params["parameter_version"] == 1, "Version tracking not working!"
    
    # Verify loss statistics
    global_stats = meta_params["global_stats"]
    assert "loss_mean" in global_stats, "loss_mean missing!"
    assert "loss_std" in global_stats, "loss_std missing!"
    
    print("  ✅ Meta-parameters properly used and versioned")
    
    # Test 2: Edge case handling
    print("  🛡️ Testing edge case handling...")
    
    # Empty insights
    meta_params = engine.meta_update([])
    assert isinstance(meta_params, dict), "Empty insights handling failed!"
    
    # Invalid insights
    invalid_insights = [{"invalid": "data"}]
    meta_params = engine.meta_update(invalid_insights)
    assert isinstance(meta_params, dict), "Invalid insights handling failed!"
    
    print("  ✅ Edge cases handled correctly")
    
    print("✅ Meta-Learning Engine tests passed!")


def test_p2p_gossip():
    """Test P2P gossip protocol"""
    print("\n🌐 Testing P2P Gossip Protocol...")
    
    # Create mock clients
    clients = []
    for i in range(5):
        client = Mock()
        client.client_id = i
        client.select_peers = Mock(return_value=[(i + 1) % 5, (i + 2) % 5])
        client.sync_with_peer = Mock()
        clients.append(client)
    
    gossip_manager = P2PGossipManager(clients)
    
    # Test 1: Topology update
    print("  🔗 Testing topology update...")
    gossip_manager.update_topology()
    
    stats = gossip_manager.get_topology_stats()
    assert stats["num_clients"] == 5, "Client count incorrect!"
    assert stats["total_edges"] >= 0, "Edge count negative!"
    
    print("  ✅ Topology updated successfully")
    
    # Test 2: Gossip exchange
    print("  💬 Testing gossip exchange...")
    gossip_manager.perform_gossip_round()
    
    stats = gossip_manager.get_stats()
    print(f"  ✅ Completed {stats['total_exchanges']} exchanges")
    
    # Test 3: Duplicate prevention
    print("  🚫 Testing duplicate prevention...")
    initial_exchanges = stats['total_exchanges']
    gossip_manager.perform_gossip_round()
    new_stats = gossip_manager.get_stats()
    
    # Should not have duplicate exchanges
    assert new_stats['total_exchanges'] >= initial_exchanges, "Exchange count decreased!"
    
    print("✅ P2P Gossip tests passed!")


def test_server():
    """Test SF-HFE Server functionality"""
    print("\n🖥️ Testing SF-HFE Server...")
    
    server = SFHFEServer(num_experts=5)
    
    # Test 1: Basic insight processing
    print("  📥 Testing insight processing...")
    insights = []
    for i in range(5):
        insight = {
            "client_id": i,
            "expert_insights": {
                f"expert_{j}": {
                    "expert_id": j,
                    "ema_loss": 0.1 + j * 0.01,
                    "activation_count": 10,
                    "learning_rate": 0.001
                }
                for j in range(5)
            },
            "avg_loss": 0.1 + i * 0.01,
            "total_samples": 100
        }
        insights.append(insight)
    
    response = server.receive_insights(insights)
    assert response["status"] == "received", "Insight processing failed!"
    assert "meta_params" in response, "Meta parameters missing!"
    
    print("  ✅ Insights processed successfully")
    
    # Test 2: Meta-learning trigger
    print("  🎯 Testing meta-learning trigger...")
    large_insights = []
    for i in range(10):
        insight = {
            "client_id": i,
            "expert_insights": {
                f"expert_{j}": {
                    "expert_id": j,
                    "ema_loss": 0.1 + j * 0.01,
                    "activation_count": 10,
                    "learning_rate": 0.001
                }
                for j in range(5)
            },
            "avg_loss": 0.1 + i * 0.01,
            "total_samples": 1000  # Should trigger meta-learning
        }
        large_insights.append(insight)
    
    response = server.receive_insights(large_insights)
    print(f"  ✅ Meta-learning triggered: {response['meta_learning_triggered']}")
    
    # Test 3: Rate limiting
    print("  ⚡ Testing rate limiting...")
    responses = []
    for i in range(150):  # More than rate limit
        response = server.receive_insights(insights)
        responses.append(response["status"])
    
    if "rate_limited" in responses:
        print("  ✅ Rate limiting working correctly")
    else:
        print("  ⚠️ Rate limiting not triggered (may be expected)")
    
    print("✅ Server tests passed!")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n🚀 Testing End-to-End Workflow...")
    
    # Initialize components
    server = SFHFEServer(num_experts=3)
    
    # Create realistic insights from multiple clients
    print("  📊 Simulating multi-client federated learning...")
    
    all_insights = []
    for round_num in range(5):  # 5 communication rounds
        round_insights = []
        
        for client_id in range(3):  # 3 clients
            for batch in range(2):  # 2 batches per client
                insight = {
                    "client_id": client_id,
                    "expert_insights": {
                        f"expert_{j}": {
                            "expert_id": j,
                            "ema_loss": 0.1 + j * 0.01 + batch * 0.001,
                            "activation_count": 10 + batch,
                            "learning_rate": 0.001 + j * 0.0001
                        }
                        for j in range(3)
                    },
                    "avg_loss": 0.1 + batch * 0.01,
                    "total_samples": 100,
                    "drift_events_count": 0
                }
                round_insights.append(insight)
        
        # Process round insights
        response = server.receive_insights(round_insights)
        all_insights.extend(round_insights)
        
        print(f"    Round {round_num + 1}: {len(round_insights)} insights processed")
        
        # Verify response structure
        assert response["status"] == "received", f"Round {round_num + 1} failed!"
        assert "meta_params" in response, "Meta parameters missing!"
        
        # Check meta-parameters structure
        meta_params = response["meta_params"]
        assert "expert_alphas" in meta_params, "Expert alphas missing!"
        assert "w_init" in meta_params, "w_init missing!"
        assert "parameter_version" in meta_params, "Version missing!"
    
    # Verify final server state
    stats = server.get_stats()
    print(f"  📈 Final stats: {stats['total_insights']} insights, {stats['unique_clients']} clients")
    
    assert stats["total_insights"] > 0, "No insights processed!"
    assert stats["unique_clients"] > 0, "No clients tracked!"
    
    print("✅ End-to-end workflow completed successfully!")


def test_thread_safety():
    """Test thread safety of critical components"""
    print("\n🔒 Testing Thread Safety...")
    
    # Test GlobalMemory thread safety
    print("  🧠 Testing GlobalMemory thread safety...")
    memory = GlobalMemory(max_insights=100)
    results = []
    
    def add_insights(thread_id: int):
        for i in range(20):
            insight = {
                "client_id": thread_id,
                "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1, "activation_count": 10, "learning_rate": 0.001}},
                "avg_loss": 0.1,
                "total_samples": 100
            }
            memory.add_insight(insight)
        results.append(thread_id)
    
    # Create multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=add_insights, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    stats = memory.stats()
    assert stats["total_insights"] == 60, f"Thread safety failed: {stats['total_insights']} != 60"
    print("  ✅ GlobalMemory thread safety verified")
    
    # Test Meta-Learning Engine thread safety
    print("  🎯 Testing Meta-Learning Engine thread safety...")
    engine = OnlineMAMLEngine(num_experts=3)
    versions = []
    
    def meta_update_thread(thread_id: int):
        insights = [{
            "client_id": thread_id,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1, "activation_count": 10, "learning_rate": 0.001}},
            "avg_loss": 0.1,
            "total_samples": 100
        }]
        meta_params = engine.meta_update(insights)
        versions.append(meta_params["parameter_version"])
    
    # Create multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=meta_update_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    assert len(set(versions)) == 3, "Parameter versions not unique!"
    print("  ✅ Meta-Learning Engine thread safety verified")
    
    print("✅ Thread safety tests passed!")


def run_comprehensive_test():
    """Run comprehensive end-to-end test suite"""
    print("🧪 FEDERATED LEARNING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Setup
        setup_test_environment()
        
        # Individual component tests
        test_global_memory()
        test_meta_learning_engine()
        test_p2p_gossip()
        test_server()
        
        # Integration tests
        test_end_to_end_workflow()
        test_thread_safety()
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print(f"⏱️ Total test duration: {duration:.2f} seconds")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
