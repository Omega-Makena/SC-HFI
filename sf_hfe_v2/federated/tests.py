"""
Comprehensive Test Suite for Federated Learning Components
Tests critical functionality including gossip synchronization, memory boundaries, and meta-learning updates
"""

import unittest
import threading
import time
import numpy as np
import torch
from typing import Dict, List
from unittest.mock import Mock, MagicMock

from .global_memory import GlobalMemory
from .meta_learning import OnlineMAMLEngine
from .gossip import P2PGossipManager
from .server import SFHFEServer
from .initialization import setup_reproducibility, initialize_system


class TestGlobalMemory(unittest.TestCase):
    """Test GlobalMemory bounded storage and thread safety"""
    
    def setUp(self):
        setup_reproducibility(42)
        self.memory = GlobalMemory(max_insights=100)
    
    def test_bounded_memory_growth(self):
        """Test that memory growth is bounded"""
        # Add more insights than the limit
        for i in range(150):
            insight = {
                "client_id": i % 10,
                "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
                "avg_loss": 0.1,
                "total_samples": 10
            }
            self.memory.add_insight(insight)
        
        # Should not exceed max_insights
        stats = self.memory.stats()
        self.assertLessEqual(stats["current_insights"], 100)
        self.assertEqual(stats["total_insights"], 150)  # Total should track all
    
    def test_insight_validation(self):
        """Test insight validation prevents corruption"""
        # Valid insight
        valid_insight = {
            "client_id": 1,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
            "avg_loss": 0.1,
            "total_samples": 10
        }
        initial_count = self.memory.total_insights
        self.memory.add_insight(valid_insight)
        self.assertEqual(self.memory.total_insights, initial_count + 1)
        
        # Invalid insight (missing field)
        invalid_insight = {
            "client_id": 2,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
            # Missing "avg_loss" and "total_samples"
        }
        initial_count = self.memory.total_insights
        self.memory.add_insight(invalid_insight)
        self.assertEqual(self.memory.total_insights, initial_count)  # Should not be added
    
    def test_thread_safety(self):
        """Test thread safety of GlobalMemory"""
        results = []
        
        def add_insights(thread_id: int):
            for i in range(50):
                insight = {
                    "client_id": thread_id,
                    "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
                    "avg_loss": 0.1,
                    "total_samples": 10
                }
                self.memory.add_insight(insight)
            results.append(thread_id)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_insights, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all insights were added safely
        stats = self.memory.stats()
        self.assertEqual(stats["total_insights"], 250)  # 5 threads * 50 insights each


class TestMetaLearningEngine(unittest.TestCase):
    """Test OnlineMAMLEngine meta-learning functionality"""
    
    def setUp(self):
        setup_reproducibility(42)
        self.engine = OnlineMAMLEngine(num_experts=5)
    
    def test_meta_parameter_usage(self):
        """Test that w_init, loss_mean, and loss_std are properly used"""
        # Create test insights
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
        
        # Perform meta update
        meta_params = self.engine.meta_update(insights)
        
        # Check that w_init is included and updated
        self.assertIn("w_init", meta_params)
        self.assertIsInstance(meta_params["w_init"], list)
        
        # Check that loss statistics are included
        global_stats = meta_params["global_stats"]
        self.assertIn("loss_mean", global_stats)
        self.assertIn("loss_std", global_stats)
        
        # Check that version tracking works
        self.assertIn("parameter_version", meta_params)
        self.assertEqual(meta_params["parameter_version"], 1)
    
    def test_edge_case_handling(self):
        """Test handling of edge cases in meta-learning"""
        # Test with empty insights
        meta_params = self.engine.meta_update([])
        self.assertIsInstance(meta_params, dict)
        
        # Test with invalid insights
        invalid_insights = [
            {"invalid": "data"},
            {"client_id": "wrong_type", "expert_insights": {}, "avg_loss": "not_a_number", "total_samples": "also_wrong"}
        ]
        meta_params = self.engine.meta_update(invalid_insights)
        self.assertIsInstance(meta_params, dict)
        
        # Test with NaN/infinity values
        nan_insights = [{
            "client_id": 1,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": float('nan'), "activation_count": float('inf'), "learning_rate": -1}},
            "avg_loss": float('nan'),
            "total_samples": 100
        }]
        meta_params = self.engine.meta_update(nan_insights)
        self.assertIsInstance(meta_params, dict)
    
    def test_thread_safety(self):
        """Test thread safety of meta-learning engine"""
        results = []
        
        def meta_update_thread(thread_id: int):
            insights = [{
                "client_id": thread_id,
                "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1, "activation_count": 10, "learning_rate": 0.001}},
                "avg_loss": 0.1,
                "total_samples": 100
            }]
            meta_params = self.engine.meta_update(insights)
            results.append(meta_params["parameter_version"])
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=meta_update_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that parameter versions are sequential
        self.assertEqual(len(set(results)), 3)  # Should have 3 different versions


class TestP2PGossipManager(unittest.TestCase):
    """Test P2P gossip protocol functionality"""
    
    def setUp(self):
        setup_reproducibility(42)
        # Create mock clients
        self.clients = []
        for i in range(5):
            client = Mock()
            client.client_id = i
            client.select_peers = Mock(return_value=[(i + 1) % 5, (i + 2) % 5])
            client.sync_with_peer = Mock()
            self.clients.append(client)
        
        self.gossip_manager = P2PGossipManager(self.clients)
    
    def test_duplicate_exchange_prevention(self):
        """Test that duplicate exchanges are prevented"""
        # Perform gossip round
        self.gossip_manager.perform_gossip_round()
        
        # Check that exchanges were recorded
        stats = self.gossip_manager.get_stats()
        self.assertGreater(stats["total_exchanges"], 0)
        
        # Check that no duplicate exchanges occurred
        exchanges = stats["recent_exchanges"]
        exchange_pairs = set()
        for exchange in exchanges:
            pair = tuple(sorted([exchange["client_a"], exchange["client_b"]]))
            self.assertNotIn(pair, exchange_pairs, "Duplicate exchange found")
            exchange_pairs.add(pair)
    
    def test_edge_counting_correctness(self):
        """Test that edge counting handles bidirectional connections correctly"""
        # Update topology
        self.gossip_manager.update_topology()
        
        # Get topology stats
        stats = self.gossip_manager.get_topology_stats()
        
        # Check that edge counting is correct
        self.assertGreaterEqual(stats["total_edges"], 0)
        self.assertLessEqual(stats["total_edges"], stats["num_clients"] * (stats["num_clients"] - 1) // 2)
    
    def test_peer_selection_uniqueness(self):
        """Test that peer selection enforces uniqueness and self-exclusion"""
        # Update topology
        self.gossip_manager.update_topology()
        
        # Check topology
        topology = self.gossip_manager.topology
        
        for client_id, peers in topology.items():
            # Check no self-references
            self.assertNotIn(client_id, peers)
            
            # Check uniqueness
            self.assertEqual(len(peers), len(set(peers)))
    
    def test_health_check_functionality(self):
        """Test client health checking"""
        # Check that all clients are initially healthy
        healthy_clients = self.gossip_manager._get_healthy_clients()
        self.assertEqual(len(healthy_clients), 5)
        
        # Simulate client becoming unhealthy
        self.gossip_manager.client_health[0]["last_seen"] = time.time() - 400  # 400 seconds ago
        healthy_clients = self.gossip_manager._get_healthy_clients()
        self.assertEqual(len(healthy_clients), 4)  # One client should be unhealthy


class TestSFHFEServer(unittest.TestCase):
    """Test SF-HFE Server functionality"""
    
    def setUp(self):
        setup_reproducibility(42)
        self.server = SFHFEServer(num_experts=5)
    
    def test_rate_limiting(self):
        """Test that rate limiting works correctly"""
        # Create test insights
        insights = [{
            "client_id": 1,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
            "avg_loss": 0.1,
            "total_samples": 100
        }]
        
        # Send many requests quickly
        responses = []
        for i in range(150):  # More than the rate limit
            response = self.server.receive_insights(insights)
            responses.append(response["status"])
        
        # Should have some rate-limited responses
        self.assertIn("rate_limited", responses)
    
    def test_meta_learning_triggers(self):
        """Test meta-learning trigger conditions"""
        # Test sample count trigger
        insights = [{
            "client_id": 1,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
            "avg_loss": 0.1,
            "total_samples": 1000  # Should trigger meta-learning
        }]
        
        response = self.server.receive_insights(insights)
        self.assertTrue(response["meta_learning_triggered"])
        
        # Test drift trigger
        insights = []
        for i in range(5):
            insights.append({
                "client_id": i,
                "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
                "avg_loss": 0.1,
                "total_samples": 100,
                "drift_events_count": 1  # All clients report drift
            })
        
        response = self.server.receive_insights(insights)
        self.assertTrue(response["meta_learning_triggered"])
    
    def test_edge_case_handling(self):
        """Test server handling of edge cases"""
        # Test with empty insights
        response = self.server.receive_insights([])
        self.assertEqual(response["status"], "error")
        
        # Test with invalid insights
        invalid_insights = [{"invalid": "data"}]
        response = self.server.receive_insights(invalid_insights)
        self.assertEqual(response["valid_insights"], 0)
        
        # Test with NaN values
        nan_insights = [{
            "client_id": 1,
            "expert_insights": {"expert_0": {"expert_id": 0, "ema_loss": 0.1}},
            "avg_loss": float('nan'),
            "total_samples": 100
        }]
        response = self.server.receive_insights(nan_insights)
        self.assertEqual(response["status"], "received")


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire system"""
    
    def setUp(self):
        initialize_system()
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from insights to meta-learning"""
        # Initialize components
        server = SFHFEServer(num_experts=3)
        
        # Create realistic insights
        insights = []
        for client_id in range(3):
            for batch in range(5):
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
                insights.append(insight)
        
        # Process insights
        response = server.receive_insights(insights)
        
        # Verify response
        self.assertEqual(response["status"], "received")
        self.assertIn("meta_params", response)
        
        # Verify meta-parameters structure
        meta_params = response["meta_params"]
        self.assertIn("expert_alphas", meta_params)
        self.assertIn("w_init", meta_params)
        self.assertIn("parameter_version", meta_params)
        
        # Verify global memory stats
        stats = server.get_stats()
        self.assertGreater(stats["total_insights"], 0)
        self.assertGreater(stats["unique_clients"], 0)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
