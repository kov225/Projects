"""
A small directed graph class with optional plotting.
"""


class VersatileDigraph:
    """A simple directed graph that can store values on nodes and weights on edges."""

    def __init__(self):
        """Start with an empty graph."""
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, value):
        """Add a node with a number attached to it."""
        if not isinstance(name, str):
            raise TypeError("Node name must be a string.")
        if not isinstance(value, (int, float)):
            raise TypeError("Node value must be numeric.")
        self.nodes[name] = value
        return True

    def add_edge(self, src, dst, edge_name=None, edge_weight=1):
        """Add a directed edge from one node to another."""
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(f"Cannot add edge, node {src} or {dst} missing.")
        if not isinstance(edge_weight, (int, float)):
            raise TypeError("Edge weight must be numeric.")
        if edge_weight < 0:
            raise ValueError("Edge weight cannot be negative.")

        if src not in self.edges:
            self.edges[src] = {}
        self.edges[src][dst] = (edge_name, edge_weight)
        return True

    def get_node_value(self, name):
        """Return the value for a node."""
        if name not in self.nodes:
            raise KeyError(f"Node {name} does not exist.")
        return self.nodes[name]

    def get_edge_weight(self, src, dst):
        """Return the weight for an edge."""
        targets = self.edges.get(src, {})
        if dst not in targets:
            raise KeyError(f"Edge {src}->{dst} does not exist.")
        return targets[dst][1]

    def predecessors(self, node):
        """Return all nodes with edges pointing to the given node."""
        if node not in self.nodes:
            raise KeyError(f"Node {node} does not exist.")
        return [src for src, targets in self.edges.items() if node in targets]

    def plot_graph(self, filename="graph"):
        """Draw the graph with Graphviz, if it’s installed."""
        try:
            from graphviz import Digraph  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                "plot_graph needs graphviz. Install with: pip install graphviz"
            ) from err

        if not self.nodes:
            raise ValueError("Graph has no nodes.")

        dot = Digraph(format="png")
        dot.attr(rankdir="LR", size="8")
        dot.attr("node", shape="ellipse", style="filled", color="lightgrey")

        for node, val in self.nodes.items():
            dot.node(node, f"{node}:{val}")

        for src, targets in self.edges.items():
            for dst, (ename, weight) in targets.items():
                label = f"{ename}:{weight}" if ename else str(weight)
                dot.edge(src, dst, label=label)

        dot.render(filename, cleanup=True)
        return f"{filename}.png"

    def plot_edge_weights(self, filename="edge_weights.html"):
        """Show edge weights as a bar chart with Bokeh, if it’s installed."""
        try:
            from bokeh.plotting import (  # pylint: disable=import-outside-toplevel
                figure,
                output_file,
                save,
            )
            from bokeh.models import HoverTool  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                "plot_edge_weights needs bokeh. Install with: pip install bokeh"
            ) from err

        if not self.edges:
            raise ValueError("Graph has no edges.")

        labels, weights = [], []
        for src, targets in self.edges.items():
            for dst, (ename, weight) in targets.items():
                labels.append(ename if ename else f"{src}->{dst}")
                weights.append(weight)

        plot = figure(
            x_range=labels,
            title="Edge Weights",
            x_axis_label="Routes",
            y_axis_label="Miles",
            plot_height=400,
            plot_width=700,
            tools="pan,box_zoom,reset,save",
        )
        plot.vbar(x=labels, top=weights, width=0.5)
        plot.add_tools(HoverTool(tooltips=[("Route", "@x"), ("Distance", "@top")]))

        output_file(filename)
        save(plot, filename)
        return filename
