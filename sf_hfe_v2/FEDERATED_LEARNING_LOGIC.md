# Federated Learning Logic in SCARCITY Framework

## 🏗️ **Core Architecture Overview**

The SCARCITY Framework implements a **privacy-preserving federated learning system** where:

- **Developer (Server)**: Has **ZERO training data** but coordinates the learning process
- **Clients**: Have their own data and train local experts
- **Communication**: Only insights are shared, never raw data

## 🔄 **Federated Learning Flow**

### **1. Initialization Phase**

```python
# Server starts with no data
server = SFHFEServer(num_experts=10)
server.global_memory = GlobalMemory(max_insights=10000)
server.meta_engine = OnlineMAMLEngine(num_experts=10)
server.gossip_manager = P2PGossipManager(clients=[]) # Will be initialized when clients are added
```

**What happens:**
- Server initializes with random meta-parameters
- Global Memory is empty (no insights yet)
- Meta-Learning Engine starts with default learning rates
- P2P Gossip Manager is ready for domain-specific learning

### **2. Client Training Phase**

```python
# Each client trains locally on their data
client = SFHFEClient(client_id="client_001", data=local_data)
client.train_experts() # Train on local data only
insights = client.extract_insights() # Extract insights, not raw data
```

**What happens:**
- Client trains experts on their local data
- Extracts insights (expert performance, loss values, sample counts)
- **Raw data never leaves the client**

### **3. Insight Sharing Phase**

```python
# Client sends insights to server
insight = {
"client_id": "client_001",
"expert_insights": {
"expert_0": {"loss": 0.25, "samples": 1000},
"expert_1": {"loss": 0.30, "samples": 800},
# ... more experts
},
"avg_loss": 0.275,
"total_samples": 1000,
"domain": "economic"
}

server.receive_insights(insight)
```

**What happens:**
- Client sends insights (not raw data) to server
- Server validates and stores insights in Global Memory
- Insights are organized by domain and client

### **4. Meta-Learning Phase**

```python
# Server runs meta-learning on insights
meta_parameters = server.meta_engine.update_meta_parameters(insights)
```

**What happens:**
- Meta-Learning Engine analyzes insights from all clients
- Updates meta-parameters (optimal initialization, learning rates)
- Learns how to better initialize experts for new clients

### **5. Domain-Specific Gossip Learning Phase**

```python
# Server coordinates P2P gossip learning within domains
server.coordinate_domain_gossip()

# Clients in the same domain exchange expert weights
for client in domain_clients:
peer_weights = get_peer_weights(client.domain)
client.gossip_exchange(peer_weights)
```

**What happens:**
- Server groups clients by domain (economic, medical, technology, etc.)
- Clients in the same domain exchange expert weights
- Expert weights are averaged within each domain
- Domain-specific knowledge is shared and improved

### **6. Parameter Broadcasting Phase**

```python
# Server broadcasts updated meta-parameters to all clients
server.broadcast_meta_parameters(meta_parameters)
```

**What happens:**
- Server sends updated meta-parameters to all clients
- Clients update their expert initialization and learning rates
- Next training round uses improved parameters

## 🧠 **Meta-Learning Logic**

### **Online MAML (Model-Agnostic Meta-Learning)**

The system uses **Online MAML** to learn optimal initialization and learning rates:

```python
class OnlineMAMLEngine:
def __init__(self, num_experts=10):
# Meta-parameters
self.w_init = torch.randn(num_experts, 64) * 0.01 # Optimal initialization
self.expert_alphas = {i: 0.01 for i in range(num_experts)} # Learning rates

def update_meta_parameters(self, insights):
# Analyze insights from all clients
for insight in insights:
expert_performance = insight["expert_insights"]
avg_loss = insight["avg_loss"]

# Update meta-parameters based on performance
self._update_initialization(expert_performance)
self._update_learning_rates(expert_performance)

return self.get_meta_parameters()
```

### **Meta-Parameter Updates**

1. **Initialization Updates**: Learn better starting weights for experts
2. **Learning Rate Updates**: Adjust learning rates based on expert performance
3. **Expert Selection**: Identify which experts work best for different domains

## 🗄️ **Global Memory Logic**

### **Insight Storage**

```python
class GlobalMemory:
def __init__(self, max_insights=10000):
self.insights = [] # Bounded storage
self.domain_partitions = defaultdict(list) # domain -> insights
self.client_insights = defaultdict(list) # client_id -> insights

def add_insight(self, insight):
# Validate insight schema
if not self._validate_insight(insight):
return

# Store insight
self.insights.append(insight)
self.domain_partitions[insight["domain"]].append(insight)
self.client_insights[insight["client_id"]].append(insight)

# Trim memory if needed
self._trim_memory()
```

### **Privacy Protection**

- **No Raw Data**: Only insights are stored, never raw data
- **Bounded Storage**: Memory is limited to prevent unlimited growth
- **Thread Safety**: Concurrent access is handled safely

## 🔄 **Communication Rounds**

### **Round Structure**

```python
class SFHFEServer:
def run_communication_round(self):
# 1. Coordinate domain-specific gossip learning
self.coordinate_domain_gossip()

# 2. Collect insights from clients
insights = self.collect_insights()

# 3. Store in Global Memory
for insight in insights:
self.global_memory.add_insight(insight)

# 4. Run Meta-Learning
meta_parameters = self.meta_engine.update_meta_parameters(insights)

# 5. Broadcast to clients
self.broadcast_meta_parameters(meta_parameters)

# 6. Update round counter
self.communication_round += 1
```

### **Round Timing**

- **Gossip Phase**: Coordinate domain-specific P2P learning
- **Collection Phase**: Collect insights from all clients
- **Processing Phase**: Run meta-learning and update parameters
- **Broadcasting Phase**: Send updated parameters to clients
- **Training Phase**: Clients train with new parameters

## **Key Features**

### **1. Privacy Preservation**
- Raw data never leaves clients
- Only insights (loss values, sample counts) are shared
- No data reconstruction possible from insights

### **2. Meta-Learning**
- Learns optimal initialization for new experts
- Adapts learning rates based on performance
- Improves over time without seeing raw data

### **3. Domain Awareness**
- Insights are organized by domain (economic, medical, etc.)
- Domain-specific meta-parameters can be learned
- Cross-domain knowledge transfer

### **4. Domain-Specific P2P Gossip Learning**
- Clients in the same domain exchange expert weights
- Domain-specific knowledge is shared and improved
- Peer selection based on domain similarity
- Asynchronous weight exchange with hysteresis-based stability

### **5. Scalability**
- Bounded memory prevents unlimited growth
- Thread-safe operations for concurrent access
- Efficient insight validation and storage

## **Example Flow**

```
1. Client A trains on economic data
→ Extracts insights: {expert_0: loss=0.25, expert_1: loss=0.30}

2. Client B trains on medical data
→ Extracts insights: {expert_0: loss=0.35, expert_2: loss=0.20}

3. Client C trains on economic data
→ Extracts insights: {expert_0: loss=0.28, expert_1: loss=0.32}

4. Server coordinates domain-specific gossip learning
→ Economic clients (A, C) exchange expert weights
→ Medical clients (B) wait for more peers

5. Server collects insights
→ Stores in Global Memory organized by domain

6. Meta-Learning Engine analyzes insights
→ Updates meta-parameters: better initialization for expert_2

7. Server broadcasts updated parameters
→ All clients receive improved meta-parameters

8. Next round: Clients train with better parameters
→ Improved performance across all clients
→ Domain-specific knowledge is shared through gossip
```

## **Benefits**

1. **Privacy**: Raw data never leaves clients
2. **Efficiency**: Only insights are communicated
3. **Learning**: Meta-learning improves over time
4. **Domain-Specific Gossip**: Clients in the same domain share knowledge directly
5. **Scalability**: Bounded memory and thread-safe operations
6. **Domain Awareness**: Domain-specific learning and transfer

This federated learning logic ensures that the Developer can coordinate learning across multiple clients while maintaining complete privacy and improving performance over time through meta-learning and domain-specific P2P gossip learning.
