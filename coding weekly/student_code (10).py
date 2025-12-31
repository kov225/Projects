"""A simple directed graph implementation with nodes and edges."""

class VersatileDigraph:
    """Directed graph where each node has a value and edges have names  weights."""

    def __init__(self):
        """Initialize an empty graph."""
        self._nodes = {}
        self._edges_by_pair = {}
        self._per_start = {}

    def _ensure_start_index(self, start_id):
        """Create helper structures for a start node if missing."""
        if start_id not in self._per_start:
            self._per_start[start_id] = {"by_name": {}, "by_end": {}}

    def _validate_node_id(self, node_id):
        """Check that the node id is a string."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string")

    def _validate_edge_name(self, edge_name):
        """Check that the edge name is a string."""
        if not isinstance(edge_name, str):
            raise TypeError("edge_name must be a string")

    def _validate_number(self, value, what):
        """Check that the value is a number."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{what} must be a number")

    def add_node(self, node_id, node_value=0):
        """Add a node or update its value."""
        self._validate_node_id(node_id)
        self._validate_number(node_value, "node_value")
        self._nodes[node_id] = node_value  # keep original type (int stays int)

    def add_edge(
        self,
        start_node_id,
        end_node_id,
        start_node_value=None,
        end_node_value=None,
        edge_name=None,
        edge_weight=0,
    ):
        """Add or update a directed edge from one node to another."""
        self._validate_node_id(start_node_id)
        self._validate_node_id(end_node_id)
        if edge_name is None:
            edge_name = f"{start_node_id}->{end_node_id}"
        self._validate_edge_name(edge_name)
        self._validate_number(edge_weight, "edge_weight")

        if start_node_id not in self._nodes:
            self._nodes[start_node_id] = 0
        if end_node_id not in self._nodes:
            self._nodes[end_node_id] = 0
        if start_node_value is not None:
            self._validate_number(start_node_value, "start_node_value")
            self._nodes[start_node_id] = start_node_value
        if end_node_value is not None:
            self._validate_number(end_node_value, "end_node_value")
            self._nodes[end_node_id] = end_node_value

        self._ensure_start_index(start_node_id)
        by_name = self._per_start[start_node_id]["by_name"]
        by_end = self._per_start[start_node_id]["by_end"]

        pair = (start_node_id, end_node_id)
        existing = self._edges_by_pair.get(pair)

        if existing is None:
            if edge_name in by_name and by_name[edge_name] != end_node_id:
                raise ValueError(
                    f'Edge name "{edge_name}" from "{start_node_id}" '
                    f'already exists to "{by_name[edge_name]}".'
                )
            self._edges_by_pair[pair] = {"weight": edge_weight, "name": edge_name}
            by_name[edge_name] = end_node_id
            by_end[end_node_id] = edge_name
        else:
            old_name = existing["name"]
            if edge_name != old_name:
                if edge_name in by_name and by_name[edge_name] != end_node_id:
                    raise ValueError(
                        f'Edge name "{edge_name}" from "{start_node_id}" '
                        f'already exists to "{by_name[edge_name]}".'
                    )
                if old_name in by_name:
                    del by_name[old_name]
                by_name[edge_name] = end_node_id
                by_end[end_node_id] = edge_name
            existing["weight"] = edge_weight
            existing["name"] = edge_name

    def get_nodes(self):
        """Return all node ids in the graph."""
        return list(self._nodes.keys())

    def get_edge_weight(self, start_node_id, end_node_id):
        """Return the weight of the edge from start to end."""
        pair = (start_node_id, end_node_id)
        if pair not in self._edges_by_pair:
            raise KeyError(f'No edge from "{start_node_id}" to "{end_node_id}".')
        return self._edges_by_pair[pair]["weight"]

    def get_edge_name(self, start_node_id, end_node_id):
        """Return the name of the edge from start to end."""
        pair = (start_node_id, end_node_id)
        if pair not in self._edges_by_pair:
            raise KeyError(f'No edge from "{start_node_id}" to "{end_node_id}".')
        return self._edges_by_pair[pair]["name"]

    def get_end_by_name(self, start_node_id, edge_name):
        """Return the end node for a given start node and edge name."""
        if start_node_id not in self._per_start:
            raise KeyError(f'No outgoing edges from "{start_node_id}".')
        by_name = self._per_start[start_node_id]["by_name"]
        if edge_name not in by_name:
            raise KeyError(f'No edge named "{edge_name}" from "{start_node_id}".')
        return by_name[edge_name]

    def get_node_value(self, node_id):
        """Return the value of a node."""
        if node_id not in self._nodes:
            raise KeyError(f'Node "{node_id}" does not exist.')
        return self._nodes[node_id]

    def print_graph(self):
        """Print all nodes and edges in plain text."""
        for node_id, node_value in self._nodes.items():
            print(f"Node {node_id} with value {node_value}")
        for (start_id, end_id), metadata in self._edges_by_pair.items():
            print(
                f"Edge from {start_id} to {end_id} "
                f"with weight {metadata['weight']} and name {metadata['name']}"
            )
