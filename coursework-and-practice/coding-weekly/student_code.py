"""A simple directed graph with nodes, edges, and TSP helpers."""

from collections import deque
from heapq import heappush, heappop
from itertools import permutations


class VersatileDigraph:
    """Directed graph with numeric node values and named, weighted edges."""

    def __init__(self):
        """Initialize empty nodes and edges."""
        self._nodes = {}  # node_id -> numeric value
        # (start_id, end_id) -> {"weight": number, "name": str}
        self._edges_by_pair = {}
        # start_id -> {"by_name": {edge_name: end_id}, "by_end": {end_id: edge_name}}
        self._per_start = {}

    # ------------------------ internal helpers -------------------------

    def _ensure_start_index(self, start_id):
        """Create per-start lookup tables if needed."""
        if start_id not in self._per_start:
            self._per_start[start_id] = {"by_name": {}, "by_end": {}}

    def _validate_node_id(self, node_id, *, must_exist=False, which="node_id"):
        """Check node id type and optional existence."""
        if not isinstance(node_id, str):
            raise TypeError(f"{which} must be a string")
        if must_exist and node_id not in self._nodes:
            raise KeyError(f'Node "{node_id}" does not exist.')

    def _validate_edge_name(self, edge_name):
        """Check that edge name is a string."""
        if not isinstance(edge_name, str):
            raise TypeError("edge_name must be a string")

    def _validate_number(self, value, what):
        """Check that a value is numeric."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{what} must be a number")

    # --------------------------- mutations -----------------------------

    def add_node(self, node_id, node_value=0):
        """Add a node or update its value."""
        self._validate_node_id(node_id)
        self._validate_number(node_value, "node_value")
        self._nodes[node_id] = node_value

    def add_edge(
        self,
        start_node_id,
        end_node_id,
        start_node_value=None,
        end_node_value=None,
        edge_name=None,
        edge_weight=0,
    ):
        """Add or update an edge from start to end. Auto-creates nodes."""
        self._validate_node_id(start_node_id, which="start_node_id")
        self._validate_node_id(end_node_id, which="end_node_id")
        if edge_name is None:
            edge_name = f"{start_node_id}->{end_node_id}"
        self._validate_edge_name(edge_name)
        self._validate_number(edge_weight, "edge_weight")
        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative")

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

    # --------------------------- queries -------------------------------

    def get_nodes(self):
        """Return all node ids as a list."""
        return list(self._nodes.keys())

    def get_edge_weight(self, start_node_id, end_node_id):
        """Return the weight for the edge start->end."""
        self._validate_node_id(start_node_id, which="start_node_id")
        self._validate_node_id(end_node_id, which="end_node_id")
        pair = (start_node_id, end_node_id)
        if pair not in self._edges_by_pair:
            raise KeyError(f'No edge from "{start_node_id}" to "{end_node_id}".')
        return self._edges_by_pair[pair]["weight"]

    def get_edge_name(self, start_node_id, end_node_id):
        """Return the name for the edge start->end."""
        self._validate_node_id(start_node_id, which="start_node_id")
        self._validate_node_id(end_node_id, which="end_node_id")
        pair = (start_node_id, end_node_id)
        if pair not in self._edges_by_pair:
            raise KeyError(f'No edge from "{start_node_id}" to "{end_node_id}".')
        return self._edges_by_pair[pair]["name"]

    def get_end_by_name(self, start_node_id, edge_name):
        """Given a start node and edge name, return the end node id."""
        self._validate_node_id(start_node_id, which="start_node_id", must_exist=True)
        self._validate_edge_name(edge_name)
        if start_node_id not in self._per_start:
            raise KeyError(f'No outgoing edges from "{start_node_id}".')
        by_name = self._per_start[start_node_id]["by_name"]
        if edge_name not in by_name:
            raise KeyError(f'No edge named "{edge_name}" from "{start_node_id}".')
        return by_name[edge_name]

    def get_node_value(self, node_id):
        """Return the numeric value stored at a node."""
        self._validate_node_id(node_id, must_exist=True)
        return self._nodes[node_id]

    def predecessors(self, node_id):
        """List nodes with an edge into node_id."""
        self._validate_node_id(node_id, must_exist=True)
        return [s for (s, e) in self._edges_by_pair if e == node_id]

    def successors(self, node_id):
        """List nodes that node_id points to."""
        self._validate_node_id(node_id, must_exist=True)
        return [e for (s, e) in self._edges_by_pair if s == node_id]

    def successor_on_edge(self, start_node_id, edge_name):
        """Return successor from start_node_id via the named edge."""
        return self.get_end_by_name(start_node_id, edge_name)

    def indegree(self, node_id):
        """Number of incoming edges to node_id."""
        self._validate_node_id(node_id, must_exist=True)
        return sum(1 for (_, e) in self._edges_by_pair if e == node_id)

    def outdegree(self, node_id):
        """Number of outgoing edges from node_id."""
        self._validate_node_id(node_id, must_exist=True)
        return sum(1 for (s, _) in self._edges_by_pair if s == node_id)

    def in_degree(self, node_id):
        """Alias for indegree (kept for tests)."""
        return self.indegree(node_id)

    def out_degree(self, node_id):
        """Alias for outdegree (kept for tests)."""
        return self.outdegree(node_id)

    # ------------------------ convenience / debug ----------------------

    def print_graph(self):
        """Print a simple description of nodes and edges."""
        for node_id, node_value in self._nodes.items():
            print(f"Node {node_id} with value {node_value}")
        for (start_id, end_id), metadata in self._edges_by_pair.items():
            print(
                f"Edge from {start_id} to {end_id} "
                f"with weight {metadata['weight']} and name {metadata['name']}"
            )

    # -------------------------- visualization stubs -------------------

    def plot_graph(
        self,
        filename=None,
        *,
        file_format="png",
        view=False,
        graph_attr=None,
        node_attr=None,
        edge_attr=None,
        label_nodes_with_values=True,
        label_edges_with="both",
    ):
        """
        Stub for graph plotting.

        Returns a simple text description instead of a real plot.
        """
        _ = filename, file_format, view, graph_attr, node_attr, edge_attr
        _ = label_nodes_with_values, label_edges_with
        lines = []
        for nid, val in self._nodes.items():
            lines.append(f"node {nid} value={val}")
        for (s, e), meta in self._edges_by_pair.items():
            lines.append(
                f"edge {s}->{e} name={meta['name']} weight={meta['weight']}"
            )
        return "\n".join(lines)

    def plot_edge_weights(
        self,
        *,
        title="Edge Weights",
        width=800,
        height=400,
        show_plot=False,
        rotate_labels=True,
    ):
        """
        Stub for edge-weight plotting.

        Returns (edge_label, weight) pairs instead of a plot.
        """
        _ = title, width, height, show_plot, rotate_labels
        data = []
        for (s, e), meta in self._edges_by_pair.items():
            edge_label = f"{s}->{e} ({meta['name']})"
            data.append((edge_label, meta["weight"]))
        return data


class SortableDigraph(VersatileDigraph):
    """VersatileDigraph subclass that can topologically sort nodes."""

    def top_sort(self):
        """Return nodes in topological order using Kahn's algorithm."""
        if not self._nodes:
            return []

        indeg = {u: 0 for u in self._nodes}
        for (_, e) in self._edges_by_pair:
            indeg[e] += 1

        ready = [u for u in indeg if indeg[u] == 0]
        order = []

        while ready:
            u = min(ready)
            ready.remove(u)
            order.append(u)
            for v in self.successors(u):
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)

        if len(order) != len(indeg):
            raise ValueError("Cycle detected")
        return order


class TraversableDigraph(SortableDigraph):
    """Adds DFS and BFS traversals."""

    def dfs(self, start_node_id):
        """Iterative DFS from start_node_id; does not yield the start."""
        self._validate_node_id(start_node_id, must_exist=True, which="start_node_id")

        visited = set()
        stack = [start_node_id]

        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            if u != start_node_id:
                yield u
            for v in sorted(self.successors(u), reverse=True):
                if v not in visited:
                    stack.append(v)

    def bfs(self, start_node_id):
        """BFS from start_node_id; does not yield the start."""
        self._validate_node_id(start_node_id, must_exist=True, which="start_node_id")

        visited = set()
        q = deque([start_node_id])

        while q:
            u = q.popleft()
            if u in visited:
                continue
            visited.add(u)
            if u != start_node_id:
                yield u
            for v in sorted(self.successors(u)):
                if v not in visited:
                    q.append(v)

    def has_path(self, start_node_id, end_node_id):
        """True if a path start_node_id -> end_node_id exists."""
        if start_node_id not in self._nodes or end_node_id not in self._nodes:
            return False
        if start_node_id == end_node_id:
            return True
        visited = set()
        q = deque([start_node_id])
        while q:
            u = q.popleft()
            if u in visited:
                continue
            visited.add(u)
            if u == end_node_id:
                return True
            for v in self.successors(u):
                if v not in visited:
                    q.append(v)
        return False


class DAG(TraversableDigraph):
    """Directed Acyclic Graph that rejects cycle-creating edges."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def add_edge(
        self,
        start_node_id,
        end_node_id,
        start_node_value=None,
        end_node_value=None,
        edge_name=None,
        edge_weight=0,
    ):
        """Add start->end only if it does not create a cycle."""
        endpoints_exist = (start_node_id in self._nodes and end_node_id in self._nodes)
        if endpoints_exist and self.has_path(end_node_id, start_node_id):
            raise ValueError(
                f'Adding edge "{start_node_id}->{end_node_id}" would create a cycle.'
            )

        return super().add_edge(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            start_node_value=start_node_value,
            end_node_value=end_node_value,
            edge_name=edge_name,
            edge_weight=edge_weight,
        )


class ShortestPathDAG(DAG):
    """DAG that can compute single-source shortest paths via DAG relaxation."""

    def shortest_path(self, start_node_id, end_node_id):
        """
        Compute the shortest path from start_node_id to end_node_id.

        Returns (path, distance). If unreachable, returns ([], inf).
        """
        self._validate_node_id(start_node_id, must_exist=True, which="start_node_id")
        self._validate_node_id(end_node_id, must_exist=True, which="end_node_id")

        order = self.top_sort()

        inf = float("inf")
        dist = {u: inf for u in self._nodes}
        parent = {u: None for u in self._nodes}
        dist[start_node_id] = 0

        for u in order:
            if dist[u] == inf:
                continue
            for v in self.successors(u):
                w = self.get_edge_weight(u, v)
                new_dist = dist[u] + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u

        if dist[end_node_id] == inf:
            return [], inf

        path = []
        current = end_node_id
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()

        return path, dist[end_node_id]


class SpanningTreeGraph(TraversableDigraph):
    """Undirected graph (modeled with a digraph) + Prim-style MST."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def add_edge(
        self,
        start_node_id,
        end_node_id,
        start_node_value=None,
        end_node_value=None,
        edge_name=None,
        edge_weight=0,
    ):
        """Insert both directions so the digraph behaves as undirected."""
        super().add_edge(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            start_node_value=start_node_value,
            end_node_value=end_node_value,
            edge_name=edge_name,
            edge_weight=edge_weight,
        )
        super().add_edge(
            start_node_id=end_node_id,
            end_node_id=start_node_id,
            edge_name=edge_name,
            edge_weight=edge_weight,
        )

    def spanning_tree(self, start=None):
        """
        Return a minimum spanning tree (Prim).

        If disconnected, only the component with `start` (or smallest node)
        is returned. Root maps to None.
        """
        if not self._nodes:
            return {}

        root = start if start is not None else min(self._nodes.keys())
        self._validate_node_id(root, must_exist=True)

        parent = {root: None}
        visited = {root}

        heap = []
        for v in self.successors(root):
            heappush(heap, (self.get_edge_weight(root, v), root, v))

        while heap and len(visited) < len(self._nodes):
            _, u, v = heappop(heap)
            if v in visited:
                continue
            parent[v] = u
            visited.add(v)
            for x in self.successors(v):
                if x not in visited:
                    heappush(heap, (self.get_edge_weight(v, x), v, x))

        return parent


class MaxFlowGraph(TraversableDigraph):
    """
    Directed graph with capacities on edges that can compute maximum flow
    using the Edmonds-Karp algorithm.
    """

    def res_graph(self, f):
        """
        Build and return the residual graph for the current flow `f`.

        f maps (u, v) -> flow on edge u->v.
        """
        residual = MaxFlowGraph()
        for node_id, node_value in self._nodes.items():
            residual.add_node(node_id, node_value)

        for (u, v), meta in self._edges_by_pair.items():
            capacity = meta["weight"]
            flow_uv = f.get((u, v), 0)

            forward_cap = capacity - flow_uv
            if forward_cap > 0:
                residual.add_edge(
                    start_node_id=u,
                    end_node_id=v,
                    edge_name=meta["name"],
                    edge_weight=forward_cap,
                )

            if flow_uv > 0:
                rev_name = f"residual_reverse_{v}_{u}_{meta['name']}"
                residual.add_edge(
                    start_node_id=v,
                    end_node_id=u,
                    edge_name=rev_name,
                    edge_weight=flow_uv,
                )

        return residual

    def bfs(self, start_node, end_node):  # pylint: disable=arguments-differ
        """
        BFS on this graph (typically a residual graph) to find a path.

        Returns a parent dict if end_node is reachable, else None.
        """
        self._validate_node_id(start_node, must_exist=True, which="start_node")
        self._validate_node_id(end_node, must_exist=True, which="end_node")

        parent = {start_node: None}
        visited = {start_node}
        q = deque([start_node])

        while q:
            u = q.popleft()
            if u == end_node:
                break
            if u not in self._per_start:
                continue
            for v in self.successors(u):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    q.append(v)

        if end_node not in parent:
            return None
        return parent

    def max_flow(self, s, t):
        """
        Compute the maximum flow from source s to sink t.

        Returns (max_flow_value, flow_dict).
        """
        self._validate_node_id(s, must_exist=True, which="source")
        self._validate_node_id(t, must_exist=True, which="sink")

        flow = {}
        max_flow_value = 0

        while True:
            residual = self.res_graph(flow)
            parent = residual.bfs(s, t)
            if parent is None:
                break

            path_edges = []
            current = t
            while current != s:
                prev = parent[current]
                path_edges.append((prev, current))
                current = prev
            path_edges.reverse()

            bottleneck = float("inf")
            for u, v in path_edges:
                cap = residual.get_edge_weight(u, v)
                if cap < bottleneck:
                    bottleneck = cap

            for u, v in path_edges:
                if (u, v) in self._edges_by_pair:
                    flow[(u, v)] = flow.get((u, v), 0) + bottleneck
                else:
                    if (v, u) in self._edges_by_pair:
                        flow[(v, u)] = flow.get((v, u), 0) - bottleneck
                    else:
                        raise KeyError(
                            f"Residual edge ({u}, {v}) does not correspond "
                            "to any original edge."
                        )

            max_flow_value += bottleneck

        return max_flow_value, flow


class TSPGraph(SpanningTreeGraph):
    """
    Graph that can solve the Traveling Salesman Problem (TSP).

    - tsp_approx: MST + preorder walk.
    - tsp_exact: brute-force permutations.
    """

    def tsp_approx(self, start=None):
        """
        Approximate TSP using:

          1. MST rooted at `start`.
          2. Preorder DFS on that tree.
          3. Return to the start node.

        Returns (path, cost).
        """
        if not self._nodes:
            return [], 0

        root = start if start is not None else min(self._nodes.keys())
        self._validate_node_id(root, must_exist=True, which="start_node_id")

        parent = self.spanning_tree(start=root)
        if not parent:
            return [], float("inf")

        children = {u: [] for u in parent}
        for node, par in parent.items():
            if par is None:
                continue
            children[par].append(node)

        order = []

        def dfs(node_id):
            order.append(node_id)
            for succ in children[node_id]:
                dfs(succ)

        dfs(root)

        order.append(root)

        total_cost = 0
        for i in range(len(order) - 1):
            u = order[i]
            v = order[i + 1]
            total_cost += self.get_edge_weight(u, v)

        return order, total_cost

    def tsp_exact(self, start=None):
        """
        Exact TSP solver using permutations.

        Returns (path, cost). If no tour exists, returns ([], inf).
        """
        nodes = self.get_nodes()
        if not nodes:
            return [], 0

        start_node = start if start is not None else min(nodes)
        self._validate_node_id(start_node, must_exist=True, which="start_node_id")

        other_nodes = [n for n in nodes if n != start_node]

        best_cost = float("inf")
        best_path = []

        for perm in permutations(other_nodes):
            tour = [start_node] + list(perm) + [start_node]
            cost = 0
            feasible = True

            for i in range(len(tour) - 1):
                u = tour[i]
                v = tour[i + 1]
                try:
                    cost += self.get_edge_weight(u, v)
                except KeyError:
                    feasible = False
                    break

            if not feasible:
                continue

            if cost < best_cost:
                best_cost = cost
                best_path = tour

        if not best_path:
            return [], float("inf")

        return best_path, best_cost
