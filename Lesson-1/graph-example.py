import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add nodes (e.g., airports)
G.add_nodes_from(["JFK", "LAX", "ATL", "ORD", "DFW"])

# Add edges (e.g., flight routes)
G.add_edges_from([("JFK", "LAX"), ("JFK", "ATL"), ("ATL", "ORD"), ("ORD", "DFW"), ("DFW", "LAX")])

# Draw the graph using nx.draw_networkx instead of nx.draw
nx.draw_networkx(G, with_labels=True, node_color="skyblue", node_size=3000, edge_color="gray", font_size=12)
plt.show()
