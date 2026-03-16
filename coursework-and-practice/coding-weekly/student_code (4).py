# pylint: disable=C

"""Graph module defining SortableDigraph, TraversableDigraph, and DAG classes."""


class SortableDigraph:
    """A directed graph that supports topological sorting."""

    def __init__(self):
        """Initialize adjacency and node value dictionaries."""
        self.adj = {}
        self.node_values = {}

    def add_node(self, node, value=None):
        """Add a node with an optional value."""
        if node not in self.adj:
            self.adj[node] = {}
        if value is not None:
            self.node_values[node] = value

    def add_edge(self, start, end, edge_weight=1):
        """Add a directed edge with optional weight."""
        if start not in self.adj:
            self.add_node(start)
        if end not in self.adj:
            self.add_node(end)
        self.adj[start][end] = edge_weight

    def get_nodes(self):
        """Return all nodes in the graph."""
        return list(self.adj.keys())

    def get_node_value(self, node):
        """Return the stored value for a given node."""
        return self.node_values.get(node)

    def get_edge_weight(self, start, end):
        """Return the weight of an edge."""
        return self.adj[start][end]

    def successors(self, node):
        """Return the direct successors of a node."""
        return list(self.adj.get(node, {}).keys())

    def predecessors(self, node):
        """Return the direct predecessors of a node."""
        preds = []
        for source, nbrs in self.adj.items():
            if node in nbrs:
                preds.append(source)
        return preds

    def top_sort(self):
        """Return nodes in topological order using Kahn’s algorithm."""
        indegree = {node: 0 for node in self.adj}
        for source in self.adj:
            for target in self.adj[source]:
                indegree[target] = indegree.get(target, 0) + 1

        # Use list as queue to avoid imports
        queue = [node for node, deg in indegree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)  # pop(0) acts like deque.popleft()
            order.append(node)
            for neighbor in self.adj.get(node, {}):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.adj):
            raise ValueError("Graph contains a cycle; topological sort invalid.")
        return order


class TraversableDigraph(SortableDigraph):
    """A directed graph supporting DFS and BFS traversal."""

    def dfs(self, start):
        """Yield nodes reachable from start using depth-first search."""
        visited = set()
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for neighbor in self.adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    yield neighbor
                    stack.append(neighbor)

    def bfs(self, start):
        """Yield nodes reachable from start using breadth-first search."""
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in self.adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    yield neighbor


class DAG(TraversableDigraph):
    """A directed acyclic graph preventing cycles on edge addition."""

    def add_edge(self, start, end, edge_weight=1):
        """Add a directed edge if it does not introduce a cycle."""
        if self._reachable(end, start):
            raise ValueError(f"Adding edge {start} → {end} would create a cycle.")
        super().add_edge(start, end, edge_weight)

    def _reachable(self, start, target):
        """Return True if target is reachable from start."""
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            for neighbor in self.adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False
