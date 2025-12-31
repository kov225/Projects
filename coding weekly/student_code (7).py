"""BinaryGraph built upon VersatileDigraph for binary tree operations."""

class VersatileDigraph:
    """A simple directed graph supporting nodes and edges."""

    def __init__(self):
        self.nodes = {}      # {node_id: value}
        self.edges = {}      # {(u, v): (edge_name, weight)}

    def add_node(self, name, value):
        """Add a node with numeric value."""
        if not isinstance(name, str):
            raise TypeError("Node name must be a string.")
        if not isinstance(value, (int, float)):
            raise TypeError("Node value must be numeric.")
        self.nodes[name] = value

    def add_edge(self, u, v, ename="", weight=0):
        """Add a directed edge from u to v."""
        if u not in self.nodes or v not in self.nodes:
            raise KeyError("Cannot add edge, node missing.")
        if not isinstance(weight, (int, float)):
            raise TypeError("Edge weight must be numeric.")
        if weight < 0:
            raise ValueError("Edge weight cannot be negative.")
        self.edges[(u, v)] = (ename, weight)

    def get_node_value(self, node):
        """Return the value of a node."""
        if node not in self.nodes:
            raise KeyError(f"Node '{node}' not found.")
        return self.nodes[node]

    def predecessors(self, node):
        """Return list of nodes that connect to this node."""
        if node not in self.nodes:
            raise KeyError(f"Node '{node}' not found.")
        return [src for (src, tgt) in self.edges if tgt == node]


class BinaryGraph(VersatileDigraph):
    """BinaryGraph inherits from VersatileDigraph to form a binary tree."""

    def __init__(self):
        super().__init__()
        self.left_child = {}
        self.right_child = {}
        # Required for tests: auto-create root
        self.add_node("Root", 0)

    def add_node_left(self, child_id, value, parent_id="Root"):
        """Add a left child node."""
        if parent_id not in self.nodes:
            raise KeyError(f"Parent '{parent_id}' not found.")
        if parent_id in self.left_child:
            raise ValueError(f"Parent '{parent_id}' already has a left child.")
        self.add_node(child_id, value)
        self.add_edge(parent_id, child_id, "L", 0)
        self.left_child[parent_id] = child_id

    def add_node_right(self, child_id, value, parent_id="Root"):
        """Add a right child node."""
        if parent_id not in self.nodes:
            raise KeyError(f"Parent '{parent_id}' not found.")
        if parent_id in self.right_child:
            raise ValueError(f"Parent '{parent_id}' already has a right child.")
        self.add_node(child_id, value)
        self.add_edge(parent_id, child_id, "R", 0)
        self.right_child[parent_id] = child_id

    def get_node_left(self, parent_id):
        """Return the left child node of a parent."""
        if parent_id not in self.left_child:
            raise KeyError(f"Parent '{parent_id}' has no left child.")
        return self.left_child[parent_id]

    def get_node_right(self, parent_id):
        """Return the right child node of a parent."""
        if parent_id not in self.right_child:
            raise KeyError(f"Parent '{parent_id}' has no right child.")
        return self.right_child[parent_id]
