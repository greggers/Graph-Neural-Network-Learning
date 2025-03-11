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

# === DRIVER CODE ===

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
