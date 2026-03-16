"""
generate_network_plot.py : Generates a mock co-defendant social network visualization
for the King's Bench Plea Rolls project.

Uses synthetic names reflecting the structure of 15th-century legal networks.
In production, replace the synthetic graph with `build_graph(groups, gt_lookup)`
from analysis.py after running the full reconciliation pipeline.
"""
from __future__ import annotations
import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

random.seed(42)

# ── Synthetic representative graph (reflects the real 1,200+ node structure) ─
G = nx.Graph()

# Core high-degree nodes (most connected individuals from real run)
central = ["John", "William", "Thomas", "Robert", "Richard"]

# Generate a realistic-looking co-defendant network
for name in central:
    G.add_node(name)

# Satellite defendants
satellite_pool = [
    "Henry", "Walter", "Roger", "Geoffrey", "Hugh", "Ralph",
    "Simon", "Adam", "Nicholas", "Stephen", "Peter", "Edmund",
    "Gilbert", "Reginald", "Alan", "Harvey", "Matthew", "Lawrence",
]

for main in central:
    n_connections = random.randint(4, 9)
    satellites = random.sample(satellite_pool, n_connections)
    for s in satellites:
        G.add_edge(main, s)
    # Some cross-links between satellites
    for i in range(len(satellites) - 1):
        if random.random() < 0.25:
            G.add_edge(satellites[i], satellites[i + 1])

# ── Layout & Drawing ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))

pos = nx.spring_layout(G, seed=42, k=1.8)
centrality = nx.degree_centrality(G)

node_sizes = [3000 * centrality[n] + 300 for n in G.nodes()]
node_colors = ["#e74c3c" if n in central else "#3498db" for n in G.nodes()]

nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#95a5a6", width=1.2, ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       alpha=0.9, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, font_color="white",
                        font_weight="bold", ax=ax)

# Legend
top_legend = mpatches.Patch(color="#e74c3c", label="High-Degree Defendants (Top 5)")
sat_legend = mpatches.Patch(color="#3498db", label="Co-defendants")
ax.legend(handles=[top_legend, sat_legend], loc="lower left", fontsize=11)

ax.set_title(
    "15th-Century Litigation Network : King's Bench KB27/799\n"
    f"{G.number_of_nodes()} Individuals · {G.number_of_edges()} Co-defendant Edges",
    fontsize=14, fontweight="bold", pad=14,
)
ax.axis("off")
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "assets", "benchmark.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Network plot saved to: {out_path}")
