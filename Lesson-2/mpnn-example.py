import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class MPNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.message_func = fn.copy_u('h', 'm')
        self.reduce_func = fn.sum('m', 'h')
        self.update_func = nn.Linear(in_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        self.update_func.reset_parameters()

    def forward(self, g, features):
        with g.local_scope():  # Avoid modifying the original graph
            g.ndata['h'] = features
            g.update_all(self.message_func, self.reduce_func)
            return self.update_func(g.ndata['h'])

# === DRIVER CODE ===

# Create a simple graph with 4 nodes and 4 directed edges
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
src, dst = zip(*edges)
g = dgl.graph((src + dst, dst + src))  # Make it undirected

# Initialize node features (4 nodes, 5 features each)
features = torch.rand((4, 5))

# Define and run the MPNN model
model = MPNNLayer(in_feats=5, out_feats=2)
output = model(g, features)

# Print the transformed node embeddings
print("Output Node Embeddings:\n", output)
