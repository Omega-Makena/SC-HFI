# SF-HFE: Scarcity Framework - Hybrid Federated Expertise

**Production-Grade Online Continual Learning System**

Train General AI with ZERO initial data through federated learning, mixture of experts, and P2P gossip protocols.

---

## Core Philosophy

**Developer has ZERO initial training data.**  
**Users have local private data.**  
**System learns from user insights only (no raw data sharing).**

---

## Clean Project Structure

```
sf_hfe_v2/                          (PRODUCTION SYSTEM)

  config.py                         - Central configuration
  main.py                           - Training orchestrator
  dashboard.py                      - Web UI (dark theme)
  
  federated/                        - FEDERATED LEARNING (Server-side)
    server.py                       - Central coordinator
    global_memory.py                - Insight storage
    meta_learning.py                - Online MAML engine
  
  moe/                              - MIXTURE OF EXPERTS (Client-side)
    client.py                       - User device
    router.py                       - Contextual bandit selector
    base_expert.py                  - Expert foundation
    
    memory/                         - 3-Tier memory system
      hierarchical.py
    
    experts/                        - 10 Specialized experts
      structure/                    - Core Structure (3)
        geometry.py                 - PCA manifolds
        temporal.py                 - LSTM sequences
        reconstruction.py           - VAE reconstruction
      
      intelligence/                 - Intelligence (2)
        causal.py                   - DAG discovery
        drift.py                    - KL divergence monitoring
      
      guardrail/                    - Guardrails (2)
        governance.py               - Constraint validation
        consistency.py              - Outlier detection
      
      specialized/                  - Specialized (3)
        peer_selection.py           - P2P topology
        meta_adaptation.py          - LR scheduling
        memory_consolidation.py     - Replay orchestration
  
  p2p/                              - P2P GOSSIP (Decentralized)
    gossip.py                       - Gossip protocol
  
  data/                             - DATA STREAMING
    stream.py                       - Concept drift generator
  
  baselines/                        - BASELINE IMPLEMENTATIONS
    fedavg.py                       - Standard FedAvg for comparison
  
  evaluation/                       - EVALUATION & FAIRNESS
    metrics.py                      - HFI and per-client metrics
```

---

## Quick Start

### Run Training
```bash
cd sf_hfe_v2
python main.py
```

### Launch Dashboard
```bash
cd sf_hfe_v2
python dashboard.py
# Dashboard opens at http://localhost:8501
```

Or use the launcher:
```bash
cd sf_hfe_v2
RUN_DASHBOARD.bat  # Windows
```

---

## Key Features

### Online Continual Learning
- No pre-training phase
- Learns from scratch on streaming data
- Single-pass through data

### 10 Specialized Experts (Organized by Category)
- Structure (3): Geometry, Temporal, Reconstruction
- Intelligence (2): Causal Inference, Drift Detection
- Guardrail (2): Governance, Statistical Consistency
- Specialized (3): Peer Selection, Meta-Adaptation, Memory

### Anti-Forgetting Mechanisms
- 3-tier hierarchical memory (Recent + Compressed + Critical)
- Experience replay with intelligent sampling
- EWC (Elastic Weight Consolidation)
- VAE compression for long-term storage

### Adaptive Meta-Learning
- Online MAML (no pre-training needed)
- Expert-specific learning rates (alpha_i)
- Trigger-based updates (drift, performance, time)

### P2P Gossip Protocol
- Decentralized weight exchange
- Adaptive topology (similarity-based)
- Hysteresis for stability

### Privacy-Preserving
- Insight-based FL (metadata only)
- No raw data/weight sharing with server
- Developer learns from insights, not data

### Research-Grade Features (NEW)
- Reproducibility: Global random seeding
- FedAvg Baseline: For comparison
- Fairness Metrics: Healthcare Fairness Index (HFI)
- Per-Client Evaluation: Individual performance tracking

---

## Statistics

| Metric | Value |
|--------|-------|
| Lines of Code | ~5,500 |
| Total Experts | 10 per client |
| Memory Tiers | 3 (Recent/Compressed/Critical) |
| Learning Mode | Pure Online (no pre-training) |
| Modules | 6 (federated/moe/p2p/data/baselines/evaluation) |
| Organization | Production-grade OOP |
| Test Coverage | 6 integration tests |

---

## Dashboard (Dark Theme)

Beautiful dark-themed web interface:
- Background: Charcoal (#121212)
- Accents: Navy Blue (#1F3A93)
- Text: Off-white (#EAEAEA)
- Cards: Graphite (#1E293B)

---

## Documentation

- `sf_hfe_v2/README.md` - Detailed module docs
- `sf_hfe_v2/STRUCTURE.md` - Organization guide
- `sf_hfe_v2/RESEARCH_COMPLIANCE.md` - Research standards compliance
- `config.py` - All configuration options
- Each module has comprehensive docstrings

---

## Usage Examples

### Import Components

```python
# Federated Learning (Server-side)
from sf_hfe_v2.federated import SFHFEServer, GlobalMemory, OnlineMAMLEngine

# MoE (Client-side)
from sf_hfe_v2.moe import SFHFEClient, ContextualBanditRouter

# Specific expert categories
from sf_hfe_v2.moe.experts.structure import GeometryExpert, TemporalExpert
from sf_hfe_v2.moe.experts.intelligence import CausalInferenceExpert
from sf_hfe_v2.moe.experts.guardrail import GovernanceExpert
from sf_hfe_v2.moe.experts.specialized import MetaAdaptationExpert

# Baseline & Evaluation
from sf_hfe_v2.baselines import FedAvgServer, FedAvgClient
from sf_hfe_v2.evaluation.metrics import compute_fairness_metrics

# P2P & Data
from sf_hfe_v2.p2p import P2PGossipManager
from sf_hfe_v2.data import ConceptDriftStream

# Reproducibility
from sf_hfe_v2.reproducibility import set_global_seed
set_global_seed(42)  # All experiments reproducible
```

### Run Simulation

```python
from sf_hfe_v2.main import OnlineTrainingOrchestrator
from sf_hfe_v2.reproducibility import set_global_seed

# Ensure reproducibility
set_global_seed(42)

# Create system
orchestrator = OnlineTrainingOrchestrator(
    num_clients=5,
    input_dim=20,
    stream_length=10000
)

# Run online training
orchestrator.run_online_training(num_batches=300)
```

---

## What Makes This Research-Grade

- Production architecture - Clean modules, proper OOP
- True online learning - No batch training or pre-training
- Sophisticated experts - LSTM, VAE, PCA, DAG discovery
- Anti-forgetting - Multi-tier memory + EWC
- Adaptive system - Dynamic LR, freeze/unfreeze
- Privacy-first - Insight-based FL (no raw data)
- Scalable design - Easy to extend and modify
- Reproducible - Global seeding for all randomness
- Fair - Healthcare Fairness Index tracking
- Comparable - FedAvg baseline included
- Well-documented - Comprehensive docstrings
- Tested - Integration test suite

---

## Next Steps

1. Run the system: `cd sf_hfe_v2 && python main.py`
2. Explore modules: Check `federated/`, `moe/`, `p2p/`, `data/`
3. Customize: Edit `config.py` for your use case
4. Add experts: Drop new experts in appropriate category folder
5. Compare baselines: Use FedAvg baseline for benchmarking
6. Track fairness: Monitor HFI and per-client metrics
7. Version 2.0: Computer Vision integration coming soon

---

## Requirements

```bash
pip install torch numpy matplotlib streamlit pandas
```

See `requirements.txt` for complete list.

---

## Developer-User Relationship

1. Developer operates server with ZERO data
2. Users join with their local data streams
3. Users train 10 experts locally (online learning)
4. Users send insights (metadata) to server
5. Server performs meta-learning from insights
6. Server broadcasts meta-parameters (alpha_i, w_init)
7. Users benefit from global knowledge without sharing data
8. P2P gossip enables fast local adaptation between similar users

---

## License

MIT License - See LICENSE file for details

---

**Built for serious AI research and production deployment**

**GitHub**: https://github.com/Omega-Makena/SC-HFI
