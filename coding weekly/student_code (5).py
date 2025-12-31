"""Module implementing a sortable directed graph with topological sorting capability."""


class VersatileDigraph:
    """A basic directed graph with adjacency-list representation."""

    def __init__(self):
        """Initialize an empty adjacency list."""
        self.adjacency = {}

    def add_node(self, node_name):
        """Add a node if it does not already exist."""
        if node_name not in self.adjacency:
            self.adjacency[node_name] = set()

    def add_edge(self, source_node, target_node):
        """Add a directed edge from source_node to target_node."""
        self.add_node(source_node)
        self.add_node(target_node)
        self.adjacency[source_node].add(target_node)

    def vertices(self):
        """Return a list of all vertex names."""
        return list(self.adjacency.keys())

    def edges(self):
        """Return a list of all directed edges as (source, target) tuples."""
        return [
            (source_node, target_node)
            for source_node, target_nodes in self.adjacency.items()
            for target_node in target_nodes
        ]


class SortableDigraph(VersatileDigraph):
    """Directed graph that supports topological sorting."""

    def top_sort(self):
        """Return vertices in topologically sorted order."""
        visited_nodes = set()
        ordered_nodes = []

        def depth_first_search(current_node):
            """Recursive DFS helper for topological sorting."""
            visited_nodes.add(current_node)
            for neighbor_node in self.adjacency.get(current_node, set()):
                if neighbor_node not in visited_nodes:
                    depth_first_search(neighbor_node)
            ordered_nodes.append(current_node)

        for node_item, _ in self.adjacency.items():
            if node_item not in visited_nodes:
                depth_first_search(node_item)

        return ordered_nodes[::-1]
