# Lesson 1: Foundations of Graphs in Machine Learning

## 1. Introduction to Graph Theory
### 1.1 What is a Graph?
A **graph** is a mathematical structure used to model relationships between entities. It consists of:
- **Nodes (Vertices)**: Represent entities (e.g., computers in a network, aircraft in air traffic).
- **Edges (Links)**: Represent connections between entities (e.g., network traffic flows, flight paths).

### 1.2 Graph Representations
Graphs can be represented in multiple ways:
- **Adjacency Matrix**: A \(N \times N\) matrix where each entry \(A_{ij}\) indicates if there is an edge between node \(i\) and node \(j\).
- **Edge List**: A list of pairs \((i, j)\) denoting an edge between node \(i\) and node \(j\).
- **Adjacency List**: A dictionary-like structure mapping nodes to their connected neighbors.

#### Example Graph Representation
Consider a simple network with three computers:
- **Nodes**: {A, B, C}
- **Edges**: {(A, B), (B, C)}

**Adjacency Matrix Representation:**
```plaintext
  A B C
A 0 1 0
B 1 0 1
C 0 1 0
```

**Adjacency List Representation:**
```python
graph = {
    "A": ["B"],
    "B": ["A", "C"],
    "C": ["B"]
}
```

## 2. Types of Graphs and Their Relevance
### 2.1 Directed vs. Undirected Graphs
- **Undirected Graph**: Edges have no direction (e.g., social networks, peer-to-peer connections).
- **Directed Graph (DiGraph)**: Edges have a direction (e.g., cybersecurity attack paths, air traffic routes).

### 2.2 Weighted vs. Unweighted Graphs
- **Unweighted Graph**: All edges are equal.
- **Weighted Graph**: Edges have weights (e.g., bandwidth in networks, fuel costs in air traffic).

### 2.3 Heterogeneous vs. Homogeneous Graphs
- **Homogeneous Graph**: All nodes and edges are of the same type.
- **Heterogeneous Graph**: Different types of nodes and edges exist (e.g., cybersecurity models with users, devices, and malicious activities).

## 3. Why Use Graphs in Machine Learning?
### 3.1 When Are Graphs Needed?
Graphs are useful when:
- **Relationships between entities matter** (e.g., network security, fraud detection).
- **Data is structured non-Euclidean** (e.g., air traffic systems).
- **Local and global interactions affect predictions** (e.g., cybersecurity threat propagation).

### 3.2 How Graphs Improve ML Models
Traditional ML methods struggle with relational data, whereas **Graph Neural Networks (GNNs)**:
- Preserve structure and connectivity.
- Leverage node neighborhoods for predictions.
- Generalize across different graph structures.

## 4. Applications in Cybersecurity and Air Traffic Management
### 4.1 Cybersecurity
- **Network Intrusion Detection**: Detect anomalous network traffic patterns.
- **Malware Propagation Analysis**: Model how threats spread across systems.
- **User Authentication Graphs**: Identify suspicious login attempts based on behavioral patterns.

#### Example: Graph-Based Anomaly Detection
Each device in a network is a **node**, and connections represent **edges**. If a new connection deviates from typical behavior (e.g., an unauthorized login attempt), it can be flagged as suspicious.

### 4.2 Air Traffic Management
- **Flight Route Optimization**: Adjust routes dynamically based on weather and congestion.
- **Delay Prediction**: Use graph-based models to predict delays by considering dependencies between flights.
- **Anomaly Detection in Air Traffic**: Identify unauthorized flight paths or unexpected deviations.

#### Example: Flight Delay Prediction
- **Nodes**: Airports
- **Edges**: Flights between airports
- **Features**: Weather, historical delays, congestion levels

## 5. Hands-on Exercise: Constructing a Graph in Python
### 5.1 Install Dependencies
```bash
pip install networkx matplotlib
```
### 5.2 Create a Simple Graph
graph-example.py
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add nodes (e.g., airports)
G.add_nodes_from(["JFK", "LAX", "ATL", "ORD", "DFW"])

# Add edges (e.g., flight routes)
G.add_edges_from([("JFK", "LAX"), ("JFK", "ATL"), ("ATL", "ORD"), ("ORD", "DFW"), ("DFW", "LAX")])

# Draw the graph
nx.draw(G, with_labels=True, node_color="skyblue", node_size=3000, edge_color="gray", font_size=12)
plt.show()
```

## 6. Summary & Next Steps
- **Graphs are powerful structures** that capture relationships in complex systems.
- **Graph-based ML models outperform traditional ML** when data is inherently relational.
- **Next Module:** Introduction to **Graph Neural Networks (GNNs)**—how to process and learn from graphs using deep learning.
