# Scarcity Framework (SF-HFE)

A Hierarchical Federated Ensemble framework for machine learning with scarcity constraints.

## Project Structure

```
scarcity/
├── core/
│   ├── __init__.py         # Core module exports
│   ├── expert.py           # Expert class - individual specialized models
│   ├── router.py           # Router class - routes inputs to experts
│   ├── client.py           # Client class - federated learning participant
│   ├── server.py           # Server class - coordinates federated learning
│   └── meta_learner.py     # MetaLearner class - meta-learning component
├── __init__.py             # Package initialization
└── run.py                  # Entry point script
```

## Stage 0 - Project Skeleton ✓

The basic project structure has been established with:
- ✓ Empty class definitions with docstrings
- ✓ Logging setup in each component
- ✓ Placeholder methods for future implementation
- ✓ Entry script with logging configuration

## Stage 2 - Federated Learning Implementation ✓

Minimal federated learning loop implemented with:
- ✓ Client owns PyTorch Linear model with local data (non-IID)
- ✓ Client.local_train() trains locally on private data
- ✓ Server.aggregate_models() implements FedAvg aggregation
- ✓ Server.broadcast_model() sends global weights to clients
- ✓ 5 global training rounds with loss convergence (33.21 → 20.02)
- ✓ No centralized data - fully federated architecture
- ✓ Comprehensive logging at each step

## Stage 3 - Scarcity-Style Insight Exchange ✓

Transformed FL into knowledge-sharing system:
- ✓ Client.generate_insight() creates structured insights instead of raw weights
  - Insight structure: `{client_id, mean_grad, uncertainty, loss_improvement, ...}`
  - Captures gradient dynamics, uncertainty scores, and learning progress
- ✓ Server.run_insight_round() orchestrates insight exchange
  - Stores all insights in `self.memory` for knowledge accumulation (25 insights)
  - No raw weight sharing - only high-level learning metadata
- ✓ MetaLearner.aggregate() processes insights
  - Computes average uncertainty across clients
  - Identifies high/low uncertainty clients for adaptive strategies
  - Maintains insight history for meta-learning
- ✓ Key Innovation: Share knowledge, not data or raw weights
- ✓ Successfully tested with 5 clients × 5 rounds = 25 insights collected

## Components

### Expert
Individual expert models that specialize in specific domains or tasks. Each expert can be trained independently and integrated into the ensemble.

### Router
Routes incoming requests to the most appropriate expert(s) using learned or rule-based strategies.

### Client
Represents a federated learning participant that holds local data and performs local training.

### Server
Coordinates the federated learning process, aggregates client updates, and manages global models.

### MetaLearner
Implements meta-learning to optimize the ensemble system across tasks and experts. In Stage 3, it aggregates structured insights from clients instead of raw model weights, identifying learning patterns and uncertainty distributions across the federation.

## Usage

Run the framework:
```bash
python scarcity/run.py
```

Or import components:
```python
from scarcity import Expert, Router, Client, Server, MetaLearner

# Initialize components
expert = Expert(expert_id=0)
router = Router(num_experts=3)
client = Client(client_id=0)
server = Server(num_clients=5)
meta_learner = MetaLearner()
```

## Development Status

- [x] Stage 0: Project skeleton and empty class definitions
- [x] Stage 2: Federated Learning implementation (FedAvg with 5 clients, 5 rounds)
- [x] Stage 3: Scarcity-style Insight Exchange (knowledge sharing without raw weights)
- [ ] Stage 4: Expert ensemble and Router integration
- [ ] Stage 5: Advanced meta-learning and optimization

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Logging

Logs are written to:
- Console (stdout)
- File: `logs/scarcity.log`

Log format: `YYYY-MM-DD HH:MM:SS - module.class - LEVEL - message`
