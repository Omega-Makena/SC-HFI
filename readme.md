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
Implements meta-learning to optimize the ensemble system across tasks and experts.

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
- [ ] Stage 3: Expert ensemble, Router, and MetaLearner integration
- [ ] Stage 4: Optimization and scaling

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Logging

Logs are written to:
- Console (stdout)
- File: `logs/scarcity.log`

Log format: `YYYY-MM-DD HH:MM:SS - module.class - LEVEL - message`
