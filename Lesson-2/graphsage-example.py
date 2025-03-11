import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv = dglnn.SAGEConv(in_dim, hidden_dim, 'mean')

    def forward(self, graph, features):
        return self.conv(graph, features)

# === DRIVER CODE ===

# Create a simple graph (4 nodes, undirected edges)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
src, dst = zip(*edges)
g = dgl.graph((src + dst, dst + src))  # Add reverse edges for undirected graph

# Number of input and hidden features
in_feats = 5
hidden_feats = 2

# Initialize random node features (4 nodes, 5 features each)
features = torch.rand((4, in_feats))

# Instantiate the GraphSAGE model
graph_sage = GraphSAGE(in_feats, hidden_feats)

# Apply the GraphSAGE layer
output = graph_sage(g, features)

# Print the output node embeddings
print("Output Node Embeddings:\n", output)
