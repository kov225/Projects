# versatile_digraph.py
"""Lightweight VersatileDigraph used for building co-defendant networks."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple
import networkx as nx


class VersatileDigraph:
    """Minimal graph class supporting node/edge management, degree centrality,
    ego networks, and conversion to NetworkX for visualization."""

    def __init__(self) -> None:
        self._adj: Dict[str, Set[str]] = {}

    def add_node(self, node: str) -> None:
        if node not in self._adj:
            self._adj[node] = set()

    def add_edge(self, u: str, v: str) -> None:
        self.add_node(u)
        self.add_node(v)
        if u != v:
            self._adj[u].add(v)
            self._adj[v].add(u)

    def nodes(self) -> List[str]:
        return list(self._adj.keys())

    def neighbors(self, node: str) -> Iterable[str]:
        return self._adj.get(node, set())

    def number_of_nodes(self) -> int:
        return len(self._adj)

    def number_of_edges(self) -> int:
        seen: Set[Tuple[str, str]] = set()
        for u, nbrs in self._adj.items():
            for v in nbrs:
                edge = (u, v) if u <= v else (v, u)
                seen.add(edge)
        return len(seen)

    def degree_centrality(self) -> Dict[str, float]:
        n = self.number_of_nodes()
        if n <= 1:
            return {node: 0.0 for node in self._adj}
        return {node: len(neighbors) / float(n - 1)
                for node, neighbors in self._adj.items()}

    def ego_subgraph(self, center: str) -> "VersatileDigraph":
        nodes = {center}
        nodes.update(self.neighbors(center))
        g = VersatileDigraph()
        for u in nodes:
            for v in self.neighbors(u):
                if v in nodes:
                    g.add_edge(u, v)
        return g

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        for node in self._adj:
            g.add_node(node)
        for u, nbrs in self._adj.items():
            for v in nbrs:
                if not g.has_edge(u, v) and u != v:
                    g.add_edge(u, v)
        return g
