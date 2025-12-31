'''Representing graphs'''


#list of sets
def part_1_graph():
    """
      a = {b, e}
      b = {c}
      c = {d, e}
      d = {b}
      e = {}
    """
    return [
        {'b', 'e'},
        {'c'},
        {'d', 'e'},
        {'b'},
        set()
    ]

#list of lists
def part_2_graph():
    """
      a = [a, b, e]
      b = [c]
      c = [a, d, e]
      d = []
      e = [d]
    """
    return [
        ['a', 'b', 'e'],
        ['c'],
        ['a', 'd', 'e'],
        [],
        ['d']
    ]

#list of dicts
def part_3_graph():
    """
      a = {a:8, b:1, e:4}
      b = {c:3}
      c = {a:2, e:4}
      d = {}
      e = {}
    """
    return [
        {'a': 8, 'b': 1, 'e': 4},
        {'c': 3},
        {'a': 2, 'e': 4},
        {},
        {}
    ]

#dict of sets
def part_4_graph():
    """
      a = {a, b, e}
      b = {c}
      c = {a}
      d = set()
      e = set()
    """
    return {
        'a': {'a', 'b', 'e'},
        'b': {'c'},
        'c': {'a'},
        'd': set(),
        'e': set()
    }

#dict of dicts
def part_5_graph():
    """
      a = {b:5}
      b = {e:3}
      c = {}
      d = {}
      e = {a:6, b:2}
    """
    return {
        'a': {'b': 5},
        'b': {'e': 3},
        'c': {},
        'd': {},
        'e': {'a': 6, 'b': 2}
    }
