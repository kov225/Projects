"""
Simple DAG (Directed Acyclic Graph) and shortest-path on a DAG.

We keep a set of nodes and, for each node, a list of outgoing edges
with weights. ShortestPathDAG then adds a method to compute the
shortest path between two nodes using topological order + relaxation.
"""


class DAG:
    """
    Directed Acyclic Graph with basic cycle protection.

    Nodes can be anything hashable (strings, numbers, etc.).
    Edges are stored as (neighbor, weight) pairs in an adjacency list.
    """

    def __init__(self):
        """Create an empty DAG with no nodes and no edges yet."""
        self.adj = {}
        self.nodes = set()

    def add_node(self, node):
        """
        Add a node to the graph if it is not already present.

        Parameters
        ----------
        node :
            The node label to add. Must be hashable.
        """
        if node not in self.nodes:
            self.nodes.add(node)
            self.adj[node] = []

    def add_edge(self, from_node, to_node, edge_weight=1):
        """
        Add a directed edge from `from_node` to `to_node`.

        If adding this edge would create a cycle, a ValueError is raised.

        Parameters
        ----------
        from_node :
            Where the edge starts.
        to_node :
            Where the edge ends.
        edge_weight : int or float, optional
            Cost / length of this edge, by default 1.

        Raises
        ------
        ValueError
            If either node does not exist, or the new edge closes a cycle.
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge.")

        # Before we actually add the edge, make sure it does not create a loop.
        if self._creates_cycle(from_node, to_node):
            raise ValueError("Edge creates a cycle in the DAG.")

        self.adj[from_node].append((to_node, edge_weight))

    def _creates_cycle(self, from_node, to_node):
        """
        Check whether an edge from `from_node` to `to_node` would form a cycle.

        The idea: if you can already reach `from_node` starting from `to_node`,
        then connecting `from_node` -> `to_node` would close a loop.
        """
        return self._dfs_reaches(to_node, from_node)

    def _dfs_reaches(self, start_node, target_node):
        """
        Return True if `target_node` can be reached from `start_node`.

        A simple depth-first search with an explicit stack is used.
        """
        visited = set()
        stack = [start_node]

        while stack:
            current = stack.pop()
            if current == target_node:
                return True
            if current not in visited:
                visited.add(current)
                # Add all neighbours to the stack to keep exploring.
                for neighbor, _ in self.adj.get(current, []):
                    stack.append(neighbor)

        return False

    def top_sort(self):
        """
        Compute a topological ordering of all nodes in the DAG.

        Returns
        -------
        list
            A list of nodes such that every edge goes
            from left to right in that list.
        """
        visited = set()
        order = []

        def depth_first_search(node):
            """
            Standard DFS: visit all children first, then record this node.

            This “post-order then reverse” trick is what gives us
            a valid topological ordering.
            """
            if node in visited:
                return
            visited.add(node)
            for neighbor, _ in self.adj[node]:
                depth_first_search(neighbor)
            order.append(node)

        # In case the graph is not fully connected, we try every node.
        for node in self.nodes:
            depth_first_search(node)

        # We built the list in reverse finish time, so flip it.
        order.reverse()
        return order


class ShortestPathDAG(DAG):
    """
    DAG that knows how to compute shortest paths.

    The algorithm is:
    1. Get a topological order.
    2. Walk through nodes in that order and relax outgoing edges.
    This only works for DAGs, but in that case it is very fast and clean.
    """

    def shortest_path(self, start_node, target_node):
        """
        Find the shortest path (and its distance) from start to target.

        Parameters
        ----------
        start_node :
            The node where we begin.
        target_node :
            The node we want to reach.

        Returns
        -------
        tuple
            (path, distance) where:
              * path is a list of nodes along the shortest route
                (empty if there is no route),
              * distance is the total weight (or float('inf') if unreachable).
        """
        # Quick win: staying where you are costs zero and needs no edges.
        if start_node == target_node:
            return [start_node], 0

        topo_order = self.top_sort()

        # Start by assuming everything is unreachable.
        distances = {node: float('inf') for node in self.nodes}
        parents = {node: None for node in self.nodes}
        distances[start_node] = 0

        # Relax edges in topological order: by the time we reach a node,
        # we already know the best distance to it so far.
        for node in topo_order:
            if distances[node] == float('inf'):
                # If we never found a way to this node, its edges
                # cannot improve anything downstream.
                continue

            for neighbor, weight in self.adj[node]:
                new_distance = distances[node] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = node

        # No route at all from start to target.
        if distances[target_node] == float('inf'):
            return [], float('inf')

        # Walk backwards from target to start using the parent pointers,
        # then reverse the list to get the path in the correct direction.
        path = []
        current = target_node
        while current is not None:
            path.append(current)
            current = parents[current]

        path.reverse()
        return path, distances[target_node]
