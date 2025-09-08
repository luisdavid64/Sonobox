import matplotlib.pyplot as plt
import networkx as nx

def visualize_topology(masses, springs, positions=None, show_labels=True):
    """
    Visualize the mass-spring topology.
    masses: list or array of mass indices (e.g., [0, 1, 2, ...])
    springs: list of tuples (i, j) indicating a spring between mass i and mass j
    positions: optional dict {i: (x, y)} for node positions; if None, uses spring layout
    show_labels: whether to show mass indices as labels
    """
    G = nx.Graph()
    G.add_nodes_from(masses)
    G.add_edges_from(springs)
    if positions is None:
        positions = nx.spring_layout(G)
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos=positions, with_labels=show_labels, node_color='skyblue', edge_color='gray', node_size=500, font_size=12)
    plt.title("Mass-Spring Topology")
    plt.show()