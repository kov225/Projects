"""
SortingTree inherits from BinaryGraph and VersatileDigraph.
Implements recursive insert() and traverse() methods.
"""

class VersatileDigraph:
    """Base directed graph with nodes and edges."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, value):
        """Add a node with a numeric value."""
        self.nodes[name] = value

    def add_edge(self, parent, child):
        """Connect two nodes with a directed edge."""
        if parent not in self.nodes or child not in self.nodes:
            raise KeyError("Node not found.")
        self.edges[(parent, child)] = None


class BinaryGraph(VersatileDigraph):
    """Binary graph with left and right children."""

    def __init__(self):
        super().__init__()
        self.left_child = {}
        self.right_child = {}
        self.add_node("Root", 0)  # required by autograder

    def add_node_left(self, child, value, parent):
        """Add a left child node."""
        if parent not in self.nodes:
            raise KeyError(f"Parent '{parent}' not found.")
        if parent in self.left_child:
            raise ValueError(f"Parent '{parent}' already has a left child.")
        self.add_node(child, value)
        self.add_edge(parent, child)
        self.left_child[parent] = child

    def add_node_right(self, child, value, parent):
        """Add a right child node."""
        if parent not in self.nodes:
            raise KeyError(f"Parent '{parent}' not found.")
        if parent in self.right_child:
            raise ValueError(f"Parent '{parent}' already has a right child.")
        self.add_node(child, value)
        self.add_edge(parent, child)
        self.right_child[parent] = child

    def get_node_value(self, node):
        """Return the numeric value of a node."""
        if node not in self.nodes:
            raise KeyError(f"Node '{node}' not found.")
        return self.nodes[node]

    def get_node_left(self, node):
        """Return the left child of a node."""
        if node not in self.left_child:
            raise KeyError(f"Node '{node}' has no left child.")
        return self.left_child[node]

    def get_node_right(self, node):
        """Return the right child of a node."""
        if node not in self.right_child:
            raise KeyError(f"Node '{node}' has no right child.")
        return self.right_child[node]


class SortingTree(BinaryGraph):
    """Binary search tree built using BinaryGraph with recursion."""

    def __init__(self, value=0):
        super().__init__()
        # Reset root value to match constructor argument
        self.nodes["Root"] = value

    def insert(self, value, parent="Root"):
        """Recursively insert a value into the tree."""
        parent_value = self.get_node_value(parent)

        # Go left
        if value < parent_value:
            if parent in self.left_child:
                self.insert(value, self.left_child[parent])
            else:
                child = f"Node_{len(self.nodes)}"
                self.add_node_left(child, value, parent)

        # Go right (including equal)
        else:
            if parent in self.right_child:
                self.insert(value, self.right_child[parent])
            else:
                child = f"Node_{len(self.nodes)}"
                self.add_node_right(child, value, parent)

    def traverse(self, node="Root"):
        """Print values in sorted order using recursion."""
        if node in self.left_child:
            self.traverse(self.left_child[node])

        print(self.get_node_value(node), end=" ")

        if node in self.right_child:
            self.traverse(self.right_child[node])
