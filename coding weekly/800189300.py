"""Robot navigation scorer."""

def robot_navigation(nodes):
    """Validate path and compute total weight."""
    graph = {
        'a': {'b': 1, 'd': 5},
        'b': {'c': 2, 'f': 5},
        'c': {'e': 1, 'h': 3},
        'd': {'f': 3},
        'e': {'d': 3, 'i': 2},
        'f': {'g': 3, 'i': 3},
        'g': {'k': 2, 'h': 2},
        'h': {'i': 1, 'j': 2, 'z': 4},
        'i': {'k': 2, 'j': 4},
        'j': {'k': 3, 'z': 4},
        'k': {'z': 3},
        'z': {}
    }

    if not isinstance(nodes, list) or not nodes or not all(isinstance(n, str) for n in nodes):
        return -1
    if nodes[0] != 'a':
        return -1
    if not all(n in graph for n in nodes):
        return -1
    if len(nodes) != len(set(nodes)):
        return -2

    total = 0
    for u, v in zip(nodes, nodes[1:]):
        if v not in graph.get(u, {}):
            return -1
        total += graph[u][v]

    last = nodes[-1]
    if last == 'z':
        return (total, last)
    return (0, last)
