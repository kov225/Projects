"""MaxFlowGraph implementation satisfying strict pylint style rules."""


class MaxFlowGraph:
    """Directed graph with Edmonds-Karp maximum flow and no external imports."""

    def __init__(self):
        """Initialise empty node set and adjacency structure."""
        self._nodes = set()
        self._edges = {}

    def add_node(self, node):
        """Add a node to the graph if it is not already present."""
        self._nodes.add(node)
        if node not in self._edges:
            self._edges[node] = {}

    def add_edge(self, u, v, edge_weight=1):
        """Add a directed edge from u to v with the given capacity."""
        self.add_node(u)
        self.add_node(v)
        self._edges[u][v] = edge_weight

    def get_nodes(self):
        """Return a list of all nodes in the graph."""
        return list(self._nodes)

    def get_neighbors(self, u):
        """Return a list of neighbors reachable from node u."""
        return list(self._edges[u].keys())

    def get_edge_weight(self, u, v):
        """Return the capacity of edge (u, v) or zero if missing."""
        if u in self._edges and v in self._edges[u]:
            return self._edges[u][v]
        return 0

    def in_degree(self, node):
        """Return the number of edges entering the given node."""
        count = 0
        for u in self._nodes:
            if node in self._edges[u]:
                count += 1
        return count

    def out_degree(self, node):
        """Return the number of edges leaving the given node."""
        return len(self._edges[node])

    def _ensure_flow_entry(self, flow, u, v):
        """Ensure flow data structure contains an entry for (u, v)."""
        if u not in flow:
            flow[u] = {}
        if v not in flow[u]:
            flow[u][v] = 0
        return flow[u][v]

    def res_graph(self, flow):
        """Construct and return the residual graph given the current flow."""
        rg = MaxFlowGraph()
        for n in self._nodes:
            rg.add_node(n)

        for u in self._nodes:
            for v in self.get_neighbors(u):
                capacity = self.get_edge_weight(u, v)
                used = self._ensure_flow_entry(flow, u, v)

                remaining = capacity - used
                if remaining > 0:
                    rg.add_edge(u, v, edge_weight=remaining)

                if used > 0:
                    rg.add_edge(v, u, edge_weight=used)

        return rg

    def bfs(self, start_node, end_node):
        """Return parent map for augmenting path using BFS."""
        visited = set()
        visited.add(start_node)
        parent = {}
        queue = [start_node]
        index = 0

        while index < len(queue):
            u = queue[index]
            index += 1

            if u == end_node:
                return parent

            for v in self.get_neighbors(u):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)

        return None

    def max_flow(self, s, t):
        """Compute and return the maximum s–t flow and the final flow map."""
        flow = {}
        total_flow = 0

        while True:
            residual = self.res_graph(flow)
            parent = residual.bfs(s, t)

            if parent is None:
                break

            bottleneck = float("inf")
            v = t
            while v != s:
                u = parent[v]
                cap = residual.get_edge_weight(u, v)
                if cap < bottleneck:
                    bottleneck = cap
                v = u

            total_flow += bottleneck

            v = t
            while v != s:
                u = parent[v]
                if v in self._edges.get(u, {}):
                    old = self._ensure_flow_entry(flow, u, v)
                    flow[u][v] = old + bottleneck
                else:
                    old = self._ensure_flow_entry(flow, v, u)
                    flow[v][u] = old - bottleneck
                v = u

        return total_flow, flow
