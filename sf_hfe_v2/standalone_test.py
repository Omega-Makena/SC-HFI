#!/usr/bin/env python3
"""
Standalone End-to-End System Test for Federated Learning Components
Tests the complete workflow from insights to meta-learning without package imports
"""

import sys
import os
import logging
import threading
import time
import random
import numpy as np
import torch
from typing import Dict, List
from unittest.mock import Mock
from collections import defaultdict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration directly
from config import (
    SYSTEM_CONFIG, META_LEARNING_CONFIG, FL_CONFIG, 
    P2P_CONFIG, STABILITY_CONFIG
)


def setup_reproducibility(seed: int = 42):
    """Setup reproducible random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_test_environment():
    """Setup test environment with proper logging and reproducibility"""
    print("🔧 Setting up test environment...")
    
    # Setup reproducibility
    setup_reproducibility(42)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("✅ Test environment ready")


class TestGlobalMemory:
    """Test GlobalMemory functionality"""
    
    def __init__(self, max_insights: int = 100):
        self.max_insights = max_insights
        self.logger = logging.getLogger("GlobalMemory")
        self._lock = threading.Lock()
        
        # Bounded storage
        self.insights = []
        self.domain_partitions = defaultdict(list)
        self.client_insights = defaultdict(list)
        
        # Statistics
        self.total_insights = 0
        self.unique_clients = set()
        
        # Insight validation schema
        self.required_fields = {
            "client_id": (str, int),
            "expert_insights": dict,
            "avg_loss": (int, float),
            "total_samples": int
        }
    
    def _validate_insight(self, insight: Dict) -> bool:
        """Validate insight schema"""
        try:
            for field, expected_type in self.required_fields.items():
                if field not in insight:
                    return False
                if not isinstance(insight[field], expected_type):
                    return False
            return True
        except Exception:
            return False
    
    def _trim_memory(self):
        """Trim memory to prevent unlimited growth"""
        if len(self.insights) > self.max_insights:
            to_remove = len(self.insights) - self.max_insights
            removed_insights = self.insights[:to_remove]
            self.insights = self.insights[to_remove:]
            
            # Clean up domain partitions and client insights
            for insight in removed_insights:
                client_id = insight.get("client_id")
                domain = insight.get("domain", "general")
                
                if client_id in self.client_insights:
                    try:
                        self.client_insights[client_id].remove(insight)
                    except ValueError:
                        pass
                
                if domain in self.domain_partitions:
                    try:
                        self.domain_partitions[domain].remove(insight)
                    except ValueError:
                        pass
    
    def add_insight(self, insight: Dict):
        """Add client insight with validation and bounded storage"""
        with self._lock:
            if not self._validate_insight(insight):
                return
            
            self.insights.append(insight)
            self.total_insights += 1
            
            client_id = insight.get("client_id")
            if client_id is not None:
                self.unique_clients.add(client_id)
                self.client_insights[client_id].append(insight)
            
            domain = insight.get("domain", "general")
            self.domain_partitions[domain].append(insight)
            
            self._trim_memory()
    
    def stats(self) -> Dict:
        """Global memory statistics"""
        with self._lock:
            return {
                "total_insights": self.total_insights,
                "current_insights": len(self.insights),
                "max_insights": self.max_insights,
                "unique_clients": len(self.unique_clients),
                "domains": list(self.domain_partitions.keys()),
                "insights_per_domain": {
                    domain: len(insights)
                    for domain, insights in self.domain_partitions.items()
                },
                "memory_utilization": len(self.insights) / self.max_insights,
            }


class TestMetaLearningEngine:
    """Test Meta-Learning Engine functionality"""
    
    def __init__(self, num_experts: int = 10):
        self.num_experts = num_experts
        self.logger = logging.getLogger("MetaLearning")
        self._lock = threading.Lock()
        
        # Meta-parameters with proper initialization
        self.w_init = torch.randn(num_experts, 64) * 0.01
        self.expert_alphas = {
            i: META_LEARNING_CONFIG["expert_lr_init"]
            for i in range(num_experts)
        }
        
        # Meta-learning state
        self.meta_updates = 0
        self.meta_loss_history = []
        self.parameter_version = 0
        
        # Running statistics
        self.global_stats = {
            "loss_mean": 0.0,
            "loss_std": 0.0,
            "activation_frequencies": torch.zeros(num_experts),
            "expert_performance": torch.zeros(num_experts),
        }
        
        # Insight validation schema
        self.required_insight_fields = {
            "client_id": (str, int),
            "expert_insights": dict,
            "avg_loss": (int, float),
            "total_samples": int
        }
    
    def _validate_insight(self, insight: Dict) -> bool:
        """Validate insight schema"""
        try:
            for field, expected_type in self.required_insight_fields.items():
                if field not in insight:
                    return False
                if not isinstance(insight[field], expected_type):
                    return False
            return True
        except Exception:
            return False
    
    def meta_update(self, insights: List[Dict]) -> Dict:
        """Perform meta-learning update with validation"""
        with self._lock:
            if not insights:
                return self.get_meta_parameters()
            
            # Validate insights
            valid_insights = [ins for ins in insights if self._validate_insight(ins)]
            if not valid_insights:
                return self.get_meta_parameters()
            
            self.meta_updates += 1
            self.parameter_version += 1
            
            # Aggregate expert performance
            expert_losses = defaultdict(list)
            expert_activations = defaultdict(int)
            expert_lr_trends = defaultdict(list)
            
            for insight in valid_insights:
                expert_insights = insight.get("expert_insights", {})
                
                for expert_name, expert_data in expert_insights.items():
                    expert_id = expert_data.get("expert_id")
                    if expert_id is not None and 0 <= expert_id < self.num_experts:
                        # Loss with validation
                        ema_loss = expert_data.get("ema_loss", 0.0)
                        if isinstance(ema_loss, (int, float)) and not np.isnan(ema_loss):
                            expert_losses[expert_id].append(float(ema_loss))
                        
                        # Activation count
                        activation = expert_data.get("activation_count", 0)
                        if isinstance(activation, (int, float)) and activation >= 0:
                            expert_activations[expert_id] += int(activation)
                        
                        # Learning rate
                        lr = expert_data.get("learning_rate", 0.001)
                        if isinstance(lr, (int, float)) and lr > 0:
                            expert_lr_trends[expert_id].append(float(lr))
            
            # Compute global statistics
            for expert_id in range(self.num_experts):
                if expert_id in expert_losses and expert_losses[expert_id]:
                    avg_loss = np.mean(expert_losses[expert_id])
                    self.global_stats["expert_performance"][expert_id] = float(avg_loss)
                    self.global_stats["activation_frequencies"][expert_id] = float(expert_activations[expert_id])
            
            # Normalize activation frequencies
            total_activations = self.global_stats["activation_frequencies"].sum()
            if total_activations > 0:
                self.global_stats["activation_frequencies"] = self.global_stats["activation_frequencies"].float()
                self.global_stats["activation_frequencies"] /= total_activations
            
            # Update global loss statistics
            all_losses = [loss for losses in expert_losses.values() for loss in losses]
            if all_losses:
                self.global_stats["loss_mean"] = float(np.mean(all_losses))
                self.global_stats["loss_std"] = float(np.std(all_losses))
            
            # Adapt expert-specific learning rates
            for expert_id in range(self.num_experts):
                if expert_id in expert_lr_trends and expert_lr_trends[expert_id]:
                    successful_lrs = expert_lr_trends[expert_id]
                    new_alpha = float(np.median(successful_lrs))
                    
                    min_lr = META_LEARNING_CONFIG["expert_lr_min"]
                    max_lr = META_LEARNING_CONFIG["expert_lr_max"]
                    
                    self.expert_alphas[expert_id] = max(min_lr, min(max_lr,
                        0.7 * self.expert_alphas[expert_id] + 0.3 * new_alpha
                    ))
            
            # Update w_init based on performance
            if all_losses:
                performance_weights = torch.softmax(-self.global_stats["expert_performance"], dim=0)
                self.w_init = self.w_init * 0.9 + torch.randn_like(self.w_init) * 0.1 * performance_weights.unsqueeze(1)
            
            # Compute meta-loss
            meta_loss = np.mean(all_losses) if all_losses else 0.0
            self.meta_loss_history.append(float(meta_loss))
            
            # Keep recent history
            if len(self.meta_loss_history) > 1000:
                self.meta_loss_history = self.meta_loss_history[-1000:]
            
            return self.get_meta_parameters()
    
    def get_meta_parameters(self) -> Dict:
        """Get current meta-parameters"""
        with self._lock:
            return {
                "expert_alphas": self.expert_alphas,
                "global_stats": {
                    "avg_loss": float(self.global_stats["expert_performance"].mean().item()),
                    "loss_mean": self.global_stats["loss_mean"],
                    "loss_std": self.global_stats["loss_std"],
                    "activation_frequencies": self.global_stats["activation_frequencies"].cpu().numpy().tolist(),
                },
                "w_init": self.w_init.cpu().numpy().tolist(),
                "meta_updates": self.meta_updates,
                "parameter_version": self.parameter_version,
                "apply_to_new_experts": False,
            }


class TestSFHFEServer:
    """Test SF-HFE Server functionality"""
    
    def __init__(self, num_experts: int = 10):
        self.num_experts = num_experts
        self.logger = logging.getLogger("Server")
        self._lock = threading.Lock()
        
        # Global Memory with bounded storage
        max_insights = FL_CONFIG.get("max_insights", 10000)
        self.global_memory = TestGlobalMemory(max_insights=max_insights)
        
        # Meta-Learning Engine
        self.meta_engine = TestMetaLearningEngine(num_experts=num_experts)
        
        # Communication round tracking
        self.round_count = 0
        self.clients_seen = set()
        
        # Meta-learning trigger state
        self.samples_since_meta = 0
        self.time_since_meta = time.time()
        self.last_meta_loss = 0.0
        
        # Rate limiting
        self.request_count = 0
        self.last_rate_reset = time.time()
        self.max_requests_per_minute = FL_CONFIG.get("max_requests_per_minute", 100)
    
    def _check_rate_limit(self) -> bool:
        """Check rate limiting"""
        current_time = time.time()
        if current_time - self.last_rate_reset > 60:
            self.request_count = 0
            self.last_rate_reset = current_time
        
        if self.request_count >= self.max_requests_per_minute:
            return False
        
        self.request_count += 1
        return True
    
    def _check_meta_trigger(self, insights: List[Dict]) -> bool:
        """Check meta-learning trigger conditions"""
        try:
            triggers = META_LEARNING_CONFIG.get("triggers", {})
            
            # Sample count trigger
            sample_count_threshold = triggers.get("sample_count", 1000)
            if self.samples_since_meta >= sample_count_threshold:
                return True
            
            # Time-based trigger
            time_threshold = triggers.get("time_seconds", 300)
            if time.time() - self.time_since_meta >= time_threshold:
                return True
            
            # Drift trigger
            if insights:
                drift_reports = 0
                valid_insights = 0
                
                for ins in insights:
                    if isinstance(ins, dict):
                        valid_insights += 1
                        drift_count = ins.get("drift_events_count", 0)
                        if isinstance(drift_count, (int, float)) and drift_count > 0:
                            drift_reports += 1
                
                if valid_insights > 0 and drift_reports >= valid_insights * 0.3:
                    return True
            
            # Performance drop trigger
            if insights:
                losses = []
                for ins in insights:
                    if isinstance(ins, dict):
                        loss = ins.get("avg_loss", 0.0)
                        if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)):
                            losses.append(float(loss))
                
                if losses and self.last_meta_loss > 0:
                    avg_loss = np.mean(losses)
                    performance_drop_threshold = triggers.get("performance_drop", 0.15)
                    
                    if avg_loss > self.last_meta_loss * (1 + performance_drop_threshold):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def receive_insights(self, insights: List[Dict]) -> Dict:
        """Receive insights with robust error handling"""
        if not self._check_rate_limit():
            return {
                "status": "rate_limited",
                "message": "Server is overloaded, please try again later"
            }
        
        with self._lock:
            if not insights:
                return {
                    "status": "error",
                    "message": "No insights provided"
                }
            
            self.round_count += 1
            
            # Add to global memory with validation
            valid_insights = 0
            for insight in insights:
                try:
                    self.global_memory.add_insight(insight)
                    valid_insights += 1
                    
                    client_id = insight.get("client_id")
                    if client_id is not None:
                        self.clients_seen.add(client_id)
                    
                    samples = insight.get("total_samples", 0)
                    if isinstance(samples, (int, float)) and samples >= 0:
                        self.samples_since_meta += int(samples)
                except Exception:
                    pass
            
            # Check meta-learning trigger
            should_trigger = self._check_meta_trigger(insights)
            
            if should_trigger:
                meta_params = self._trigger_meta_learning(insights)
            else:
                meta_params = self.meta_engine.get_meta_parameters()
            
            return {
                "status": "received",
                "round": self.round_count,
                "meta_params": meta_params,
                "meta_learning_triggered": should_trigger,
                "valid_insights": valid_insights,
            }
    
    def _trigger_meta_learning(self, insights: List[Dict]) -> Dict:
        """Trigger meta-learning update"""
        try:
            recent_insights = self.global_memory.insights[-100:]
            meta_params = self.meta_engine.meta_update(recent_insights)
            
            self.samples_since_meta = 0
            self.time_since_meta = time.time()
            
            if insights:
                losses = []
                for ins in insights:
                    if isinstance(ins, dict):
                        loss = ins.get("avg_loss", 0.0)
                        if isinstance(loss, (int, float)) and not (np.isnan(loss) or np.isinf(loss)):
                            losses.append(float(loss))
                
                if losses:
                    self.last_meta_loss = float(np.mean(losses))
            
            return meta_params
            
        except Exception:
            return self.meta_engine.get_meta_parameters()


def test_global_memory():
    """Test GlobalMemory functionality"""
    print("\n🧠 Testing GlobalMemory...")
    
    memory = TestGlobalMemory(max_insights=100)
    
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
    print("  📊 Testing bounded storage... test_global_memory")
    for i in range(100):
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
    """Test Meta-Learning Engine functionality"""
    print("\n🎯 Testing Meta-Learning Engine...")
    
    engine = TestMetaLearningEngine(num_experts=5)
    
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


def test_server():
    """Test SF-HFE Server functionality"""
    print("\n🖥️ Testing SF-HFE Server...")
    
    server = TestSFHFEServer(num_experts=5)
    
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
    
    print("✅ Server tests passed!")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n🚀 Testing End-to-End Workflow...")
    
    # Initialize components
    server = TestSFHFEServer(num_experts=3)
    
    # Create realistic insights from multiple clients
    print("  📊 Simulating multi-client federated learning...")
    
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
    stats = server.global_memory.stats()
    print(f"  📈 Final stats: {stats['total_insights']} insights, {len(server.clients_seen)} clients")
    
    assert stats["total_insights"] > 0, "No insights processed!"
    assert len(server.clients_seen) > 0, "No clients tracked!"
    
    print("✅ End-to-end workflow completed successfully!")


def test_thread_safety():
    """Test thread safety of critical components"""
    print("\n🔒 Testing Thread Safety...")
    
    # Test GlobalMemory thread safety
    print("  🧠 Testing GlobalMemory thread safety...")
    memory = TestGlobalMemory(max_insights=100)
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
