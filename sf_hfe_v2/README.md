# SF-HFE v2.0 - Production-Grade Online Continual Learning

**Scarcity Framework: Hybrid Federated Expertise**

Clean, modular architecture for building General AI with ZERO initial data.

---

## Organized Structure (Easy to Navigate)

```
sf_hfe_v2/
  
  config.py                           - Central configuration
  main.py                             - Main orchestrator
  README.md                           - This file
  STRUCTURE.md                        - Detailed organization docs
  
  federated/                          - FEDERATED LEARNING
    server.py                         - Server coordinator
    global_memory.py                  - Insight storage
    meta_learning.py                  - Online MAML engine
    __init__.py
  
  moe/                                - MIXTURE OF EXPERTS
    client.py                         - User device
    router.py                         - Expert selector
    base_expert.py                    - Expert base class
    __init__.py
    
    memory/                           - 3-Tier Memory
      hierarchical.py                 - Main interface
      __init__.py
    
    experts/                          - 10 Experts (Organized)
      __init__.py
      
      structure/                      - Core Structure (3)
        geometry.py                   - PCA manifolds
        temporal.py                   - LSTM sequences
        reconstruction.py             - VAE reconstruction
        __init__.py
      
      intelligence/                   - Intelligence (2)
        causal.py                     - DAG discovery
        drift.py                      - KL divergence
        __init__.py
      
      guardrail/                      - Guardrails (2)
        governance.py                 - Constraints
        consistency.py                - Outlier detection
        __init__.py
      
      specialized/                    - Specialized (3)
        peer_selection.py             - P2P topology
        meta_adaptation.py            - LR scheduling
        memory_consolidation.py       - Replay orchestration
        __init__.py
  
  p2p/                                - P2P GOSSIP
    gossip.py                         - Gossip protocol
    __init__.py
  
  data/                               - DATA STREAMING
    stream.py                         - Concept drift generator
    __init__.py
  
  baselines/                          - BASELINE IMPLEMENTATIONS
    fedavg.py                         - Standard FedAvg
    __init__.py
  
  evaluation/                         - EVALUATION & FAIRNESS
    metrics.py                        - HFI and per-client metrics
    __init__.py
```

---

## What Makes This Organized:

### 1. **Clear Separation**
   - **federated/** = Server-side only
   - **moe/** = Client-side only
   - **p2p/** = Decentralized communication
   - **data/** = Input/output

### 2. **Expert Categories**
   - **structure/** = Data structure & patterns (3 experts)
   - **intelligence/** = Analysis & monitoring (2 experts)
   - **guardrail/** = Safety & validation (2 experts)
   - **specialized/** = System optimization (3 experts)

### 3. **Logical Grouping**
   - Related code stays together
   - Easy to find what you need
   - Better IDE navigation
   - Scales well for new features

---

## Quick Start

```bash
cd sf_hfe_v2
python main.py
```

---

## Module Guide

### **federated/** - Server Components
Developer operates this with **ZERO data**:
- `server.py` - Main server coordinator
- `global_memory.py` - Stores client insights
- `meta_learning.py` - MAML engine (learns from metadata)

### **moe/** - MoE Components
Users run this with their **local data**:
- `client.py` - User device with 10 experts
- `router.py` - Contextual bandit selector
- `base_expert.py` - Expert foundation
- `memory/` - 3-tier anti-forgetting system
- `experts/` - 10 specialized experts (categorized!)

### **p2p/** - P2P Communication
- `gossip.py` - Decentralized weight exchange

### **data/** - Data Streaming
- `stream.py` - Synthetic streams with concept drift

---

## Import Examples

```python
# Import server (Developer side)
from federated import SFHFEServer, GlobalMemory, OnlineMAMLEngine

# Import client (User side)
from moe import SFHFEClient, ContextualBanditRouter

# Import specific expert category
from moe.experts.structure import GeometryExpert, TemporalExpert
from moe.experts.intelligence import CausalInferenceExpert
from moe.experts.guardrail import GovernanceExpert
from moe.experts.specialized import MetaAdaptationExpert

# Import P2P
from p2p import P2PGossipManager

# Import data streaming
from data import ConceptDriftStream
```

---

## Production-Grade Features

- 4,500+ lines of real code  
- 10 specialized experts with unique architectures  
- 3-tier memory (Recent + Compressed + Critical)  
- Online MAML meta-learning  
- EWC anti-forgetting  
- UCB router with entropy regularization  
- P2P gossip with adaptive topology  
- Modular & extensible  

---

## Documentation

- `STRUCTURE.md` - Detailed organization & file mapping
- `config.py` - All configuration options
- Each module has docstrings

---

## Next Steps

1. Run the system: `python main.py`
2. Explore expert categories in `moe/experts/`
3. Customize config in `config.py`
4. Add new experts in appropriate category folders
5. Build Version 2.0 with Computer Vision!

---

**Clean, organized, production-ready!**
