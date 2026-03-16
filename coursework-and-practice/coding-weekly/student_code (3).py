"""Student graph implementation for Prim's algorithm assignment."""


class VersatileDigraph:
    """A simple directed graph with weighted edges."""

    def __init__(self):
        """Create an empty directed graph."""
        self.graph = {}

    def add_node(self, node):
        """Add a node to the graph if it does not already exist."""
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, start_node, end_node, edge_weight=1):
        """Add a directed edge from start_node to end_node with a weight."""
        if start_node not in self.graph:
            self.add_node(start_node)
        if end_node not in self.graph:
            self.add_node(end_node)
        self.graph[start_node][end_node] = edge_weight

    def get_edge_weight(self, start_node, end_node):
        """
        Return the weight of the edge from start_node to end_node.

        If the edge does not exist, return None.
        """
        if start_node in self.graph and end_node in self.graph[start_node]:
            return self.graph[start_node][end_node]
        return None

    def neighbors(self, node):
        """Return a list of neighbors for the given node."""
        if node in self.graph:
            return list(self.graph[node].keys())
        return []

    def get_nodes(self):
        """Return a list of all nodes in the graph."""
        return list(self.graph.keys())

    def __contains__(self, node):
        """Return True if the node exists in the graph."""
        return node in self.graph

    def __str__(self):
        """Return a string representation of the adjacency structure."""
        return str(self.graph)


class SpanningTreeGraph(VersatileDigraph):
    """
    An undirected graph implemented on top of VersatileDigraph.

    The add_edge method adds edges in both directions,
    and spanning_tree computes a minimum spanning tree using Prim's algorithm.
    """

    def add_edge(self, start_node, end_node, edge_weight=1):
        """Add an undirected edge by inserting both directions."""
        super().add_edge(start_node, end_node, edge_weight=edge_weight)
        super().add_edge(end_node, start_node, edge_weight=edge_weight)

    def spanning_tree(self, start_node):
        """
        Compute a minimum spanning tree using Prim's algorithm.

        Returns a dictionary mapping each reachable node to its parent in the
        spanning tree. The start_node has parent None. Nodes that are not
        reachable from start_node are not included.
        """
        if start_node not in self.graph:
            return {}

        visited = set([start_node])
        parent = {start_node: None}

        edges = []
        for neighbor, weight in self.graph[start_node].items():
            edges.append((weight, start_node, neighbor))

        while edges:
            # select the smallest-weight edge manually (no imports)
            min_edge = edges[0]
            for edge in edges[1:]:
                if edge[0] < min_edge[0]:
                    min_edge = edge
            edges.remove(min_edge)

            weight, from_node, to_node = min_edge
            if to_node in visited:
                continue

            visited.add(to_node)
            parent[to_node] = from_node

            for neighbor, wgt in self.graph[to_node].items():
                if neighbor not in visited:
                    edges.append((wgt, to_node, neighbor))

        return parent

    def dfs(self, start_node):
        """Perform a depth-first search starting at start_node."""
        if start_node not in self.graph:
            return []
        visited = []
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                # reversed for deterministic order
                stack.extend(reversed(sorted(self.graph[node].keys())))
        return visited

    def bfs(self, start_node):
        """Perform a breadth-first search starting at start_node."""
        if start_node not in self.graph:
            return []
        visited = []
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                queue.extend(sorted(self.graph[node].keys()))
        return visited
