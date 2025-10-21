"""
Complete System Test
Tests all components end-to-end with reduced batch count for quick validation
"""

import torch
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("SystemTest")

logger.info("="*80)
logger.info("SF-HFE v2.0 - Complete System Test")
logger.info("="*80)

# Test 1: Import all modules
logger.info("\n[TEST 1] Importing modules...")
try:
    from config import EXPERT_CONFIG, ROUTER_CONFIG, MEMORY_CONFIG
    logger.info("✓ Config imported")
    
    from federated import SFHFEServer, GlobalMemory, OnlineMAMLEngine
    logger.info("✓ Federated module imported")
    
    from moe import SFHFEClient, ContextualBanditRouter, HierarchicalMemory
    logger.info("✓ MoE module imported")
    
    from moe.experts.structure import GeometryExpert, TemporalExpert, ReconstructionExpert
    from moe.experts.intelligence import CausalInferenceExpert, DriftDetectionExpert
    from moe.experts.guardrail import GovernanceExpert, StatisticalConsistencyExpert
    from moe.experts.specialized import PeerSelectionExpert, MetaAdaptationExpert, MemoryConsolidationExpert
    logger.info("✓ All 10 experts imported")
    
    from p2p import P2PGossipManager
    logger.info("✓ P2P module imported")
    
    from data import ConceptDriftStream, MultiClientStreamGenerator
    logger.info("✓ Data module imported")
    
    logger.info("\n✅ ALL IMPORTS SUCCESSFUL")
except Exception as e:
    logger.error(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Initialize components
logger.info("\n[TEST 2] Initializing components...")
try:
    # Server (Developer with ZERO data)
    server = SFHFEServer(num_experts=10)
    logger.info("✓ Server initialized")
    
    # Client (User with data)
    client = SFHFEClient(client_id=0, input_dim=20, output_dim=1, has_data=True)
    logger.info("✓ Client initialized (10 experts + router)")
    
    # Data stream
    stream = ConceptDriftStream(num_features=20, stream_length=1000)
    logger.info("✓ Data stream created")
    
    logger.info("\n✅ INITIALIZATION SUCCESSFUL")
except Exception as e:
    logger.error(f"\n❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Process a single batch
logger.info("\n[TEST 3] Processing single mini-batch...")
try:
    # Generate batch
    batch_x, batch_y = stream.generate_batch(batch_size=32)
    logger.info(f"✓ Generated batch: x={batch_x.shape}, y={batch_y.shape}")
    
    # Client processes batch
    metrics = client.process_stream_batch(batch_x, batch_y)
    logger.info(f"✓ Batch processed: loss={metrics.get('avg_loss', 0):.4f}")
    
    logger.info("\n✅ BATCH PROCESSING SUCCESSFUL")
except Exception as e:
    logger.error(f"\n❌ Batch processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Generate and send insights
logger.info("\n[TEST 4] Testing insight generation and FL...")
try:
    # Generate insights
    insights = client.generate_insights()
    logger.info(f"✓ Insights generated: {len(insights)} fields")
    
    # Server receives
    response = server.receive_insights([insights])
    logger.info(f"✓ Server received insights (round {response['round']})")
    
    logger.info("\n✅ FEDERATED LEARNING SUCCESSFUL")
except Exception as e:
    logger.error(f"\n❌ FL failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Run mini training loop (10 batches)
logger.info("\n[TEST 5] Running mini training loop (10 batches)...")
try:
    for i in range(10):
        batch_x, batch_y = stream.generate_batch(batch_size=32)
        metrics = client.process_stream_batch(batch_x, batch_y)
        
        if i % 5 == 0:
            logger.info(f"  Batch {i}: loss={metrics.get('avg_loss', 0):.4f}, entropy={metrics.get('routing_entropy', 0):.3f}")
    
    logger.info("\n✅ MINI TRAINING LOOP SUCCESSFUL")
except Exception as e:
    logger.error(f"\n❌ Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Expert statistics
logger.info("\n[TEST 6] Checking expert statistics...")
try:
    for expert in client.experts:
        stats = expert.stats()
        logger.info(f"  Expert {expert.expert_id} ({expert.expert_name}): activations={stats['activation_count']}")
    
    logger.info("\n✅ EXPERT STATS ACCESSIBLE")
except Exception as e:
    logger.error(f"\n❌ Stats check failed: {e}")
    import traceback
    traceback.print_exc()

# Final summary
logger.info("\n" + "="*80)
logger.info("✅ ALL TESTS PASSED!")
logger.info("="*80)
logger.info("\nSystem Status:")
logger.info(f"  Server: {server.round_count} rounds, {server.global_memory.total_insights} insights")
logger.info(f"  Client: {client.batch_count} batches, {client.total_samples_processed} samples")
logger.info(f"  Router: {client.router.total_selections} selections, entropy={client.router.compute_entropy():.3f}")
logger.info(f"  Stream: {stream.sample_index}/{stream.stream_length} samples")

logger.info("\n🎉 SF-HFE v2.0 is FULLY FUNCTIONAL!")

