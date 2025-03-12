# Lesson 2: Graph Neural Network Architectures  

## Introduction  

Graph Neural Networks (**GNNs**) are designed to process **graph-structured data**, capturing relationships between nodes using **message passing** techniques.  
Unlike traditional deep learning models, which operate on Euclidean data (e.g., images, text), GNNs generalize deep learning methods to **non-Euclidean data**.  

This lesson introduces the **core GNN architectures**, including:  
**Message Passing Neural Networks (MPNNs)**  
**Graph Convolutional Networks (GCNs)**  
**Graph Attention Networks (GATs)**  
**GraphSAGE (Sampling-Based GNNs)**  

---

## 1. How Do GNNs Work?  

At their core, GNNs follow a **message passing** paradigm:  

1. **Aggregation**: Nodes collect information from their neighbors.  
2. **Update**: Node representations are updated using aggregated information.  
3. **Propagation**: Updated representations are passed to the next layer.  
4. **Readout**: A final node or graph-level representation is computed.  

The **general update equation** for a node $v$ is:  
$$
h_v^{(k+1)} = \sigma \left( W \cdot \text{AGG} \left( \{ h_u^{(k)} | u \in \mathcal{N}(v) \} \right) \right)
$$  
where:  
- $h_v^{(k)}$ is the hidden state of node $v$ at layer $k$.  
- $\text{AGG}(\cdot)$ is an aggregation function (sum, mean, max, attention).  
- $W$ is a learnable weight matrix.  
- $\sigma$ is a non-linear activation (e.g., ReLU).  

---
## 2. Popular GNN Architectures  

### **2.1 Graph Convolutional Networks (GCN)**  

GCNs extend the concept of **convolution** from CNNs to graphs. Instead of applying filters on grid-structured data, GCNs aggregate features from neighboring nodes.  

#### **GCN Layer Equation**  
$$
H^{(k+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)} \right)
$$  
where:  
- $\tilde{A} = A + I$ (adjacency matrix with self-loops).  
- $\tilde{D}$ is the degree matrix.  
- $H^{(k)}$ is the node feature matrix at layer $ k $.  

#### **GCN in PyTorch (DGL)**  
```python
import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, graph, features):
        with graph.local_scope():
            graph.ndata['h'] = features
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            h = graph.ndata['h']
            return self.linear(h)
```

---

### **2.2 Graph Attention Networks (GAT)**  

Instead of equally weighting all neighbors, GATs **learn importance scores** between nodes using an **attention mechanism**.  

#### **GAT Attention Mechanism**  
$$
\alpha_{ij} = \frac{\exp\left( \text{LeakyReLU} \left( a^T [W h_i || W h_j] \right) \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \text{LeakyReLU} \left( a^T [W h_i || W h_k] \right) \right)}
$$ 
where $\alpha_{ij}$ represents the attention score between node $i$ and node $j$.  

#### **GAT in PyTorch (DGL)**  
```python
import dgl.nn as dglnn

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super().__init__()
        self.conv = dglnn.GATConv(in_dim, hidden_dim, num_heads=num_heads)

    def forward(self, graph, features):
        return self.conv(graph, features).mean(1)  # Aggregate over heads
```

---

### **2.3 GraphSAGE (Sampling-Based GNN)**  

GraphSAGE improves scalability by **sampling** a subset of neighbors rather than aggregating all of them.  

#### **GraphSAGE Aggregation**  
$$
h_v^{(k+1)} = \sigma \left( W \cdot \text{AGG} \left( \{ h_u^{(k)} | u \in S_{\mathcal{N}(v)} \} \right) \right)
$$ 
where $S_{\mathcal{N}(v)}$ is a sampled subset of neighbors.  

#### **GraphSAGE in PyTorch (DGL)**  
```python
import dgl.nn.pytorch as dglnn

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv = dglnn.SAGEConv(in_dim, hidden_dim, 'mean')

    def forward(self, graph, features):
        return self.conv(graph, features)
```

---

## 3. Hands-on Exercise: Build a Simple GCN  

### **3.1 Install Dependencies**  
```bash
pip install dgl torch matplotlib
```

### **3.2 Load a Sample Graph**  
```python
import dgl
import torch

# Create a simple graph (4 nodes, undirected edges)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
src, dst = zip(*edges)
g = dgl.graph((src + dst, dst + src))  # Add reverse edges for undirected graph

# Number of input and output features
in_feats = 5
out_feats = 2

# Initialize random node features (4 nodes, 5 features each)
features = torch.rand((4, in_feats))

# Instantiate the GCN layer
gcn_layer = GCNLayer(in_feats, out_feats)

# Apply the GCN layer
output = gcn_layer(g, features)

# Print the output node embeddings
print("Output Node Embeddings:\n", output)
```

---

## 4. Summary & Next Steps  

### **Key Takeaways**  
- **GNNs use message passing** to learn node embeddings.  
- **GCN** aggregates neighbor features using adjacency matrices.  
- **GAT** applies **self-attention** to assign importance to neighbors.  
- **GraphSAGE** scales GNNs by **sampling** neighbor nodes.  

### **Next Lesson:** **Graph Convolutional Networks (GCN) - Theory & Implementation**  

---

**Questions or Ideas?** Open an issue or contribute to this repository!  

---
**Author:** Greg Wagner, PhD | *[gregorymwagner.com](http://www.gregorymwagner.com)*  
