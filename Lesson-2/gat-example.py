import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super().__init__()
        self.conv = dglnn.GATConv(in_dim, hidden_dim, num_heads=num_heads)

    def forward(self, graph, features):
        return self.conv(graph, features).mean(1)  # Aggregate over attention heads

# === DRIVER CODE ===

# Create a simple graph (4 nodes, undirected edges)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
src, dst = zip(*edges)
g = dgl.graph((src + dst, dst + src))  # Add reverse edges for undirected graph

# Number of input features, hidden dimensions, and attention heads
in_feats = 5
hidden_feats = 2
num_heads = 3  # Multi-head attention

# Initialize random node features (4 nodes, 5 features each)
features = torch.rand((4, in_feats))

# Instantiate the GAT model
gat_model = GAT(in_feats, hidden_feats, num_heads)

# Apply the GAT layer
output = gat_model(g, features)

# Print the output node embeddings
print("Output Node Embeddings:\n", output)
