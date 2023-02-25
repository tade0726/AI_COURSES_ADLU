import sys
from collections import deque
import numpy as np

import functools
import heapq


test = False

### --- utils ---


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:

        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val

    else:

        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order="min", f=lambda x: x):
        self.heap = []
        if order == "min":
            self.f = f
        elif order == "max":  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item, extra_order: int = 0):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, ((self.f(item), extra_order), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Trying to pop from empty PriorityQueue.")

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)


### --- utils ---


class Problem:
    """The abstract class for a formal problem."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self, state):
        raise NotImplementedError


class Node:
    """A node in a search tree."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [
            self.child_node(problem, action) for action in problem.actions(self.state)
        ]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(
            next_state,
            self,
            action,
            problem.path_cost(self.path_cost, self.state, action, next_state),
        )
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def state_solutions(self):
        """Return the sequence of states to go from the root to this node."""
        return [node.state for node in self.path()]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


class Map(Problem):
    def __init__(self, initial, goal, map_matrix: np.array):
        super().__init__(initial, goal)
        self.matrix = map_matrix
        self.matrix_shape = map_matrix.shape
        self.visited_matrix = np.ones_like(self.matrix) * False

    def legal_actions(self, action):

        # out of boundary
        if (
            action[0] < 0
            or action[0] + 1 > self.matrix_shape[0]
            or action[1] < 0
            or action[1] + 1 > self.matrix_shape[1]
        ):
            return False

        elif self.matrix[action] == np.inf:
            return False
        else:
            return True

    def actions(self, state):
        m, n = state
        next_actions = [
            (m - 1, n),
            (m + 1, n),
            (m, n - 1),
            (m, n + 1),
        ]  # up, down, left. right
        actions_coordinates = []
        for i in next_actions:
            if self.legal_actions(i):
                actions_coordinates.append(i)
        return actions_coordinates

    def result(self, state, action):
        # the result is the action (coordinates)
        return action

    def value(self, state):
        return self.matrix[state]

    def path_cost(self, c, state1, action, state2):
        extra_cost = self.matrix[state2] - self.matrix[state1]
        extra_cost = extra_cost if extra_cost > 0 else 0
        return c + 1 + extra_cost

    def manhattan_h(self, n: Node):
        dx = abs(self.goal[0] - n.state[0])
        dy = abs(self.goal[1] - n.state[1])
        D = 1
        return D * (dx + dy)

    def euclidean_h(self, n: Node):
        dx = abs(self.goal[0] - n.state[0])
        dy = abs(self.goal[1] - n.state[1])
        D = 1
        return D * np.sqrt(dx**2 + dy**2)


def parse_map_to_matrix(lines: str):
    def float_list(numbers_in_str):
        return [float(i) if i != "X" else np.inf for i in numbers_in_str]

    def int_list(numbers_in_str):
        return [int(i) for i in numbers_in_str]

    m = []

    lines = [line.strip("\n") for line in lines]

    start = int_list(lines[1].split(" "))
    end = int_list(lines[2].split(" "))

    start = (start[0] - 1, start[1] - 1)
    end = (end[0] - 1, end[1] - 1)

    for line in lines[3:]:
        ns = float_list(line.split(" "))
        m.append(ns)

    return start, end, np.array(m)


def breadth_first_tree_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    frontier = deque([Node(problem.initial)])  # FIFO queue
    explored = set()

    while frontier:
        node = frontier.popleft()

        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored:
                frontier.append(child)
    return None


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first."""
    f = memoize(f, "f")
    node = Node(problem.initial)
    frontier = PriorityQueue("min", f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(
                    len(explored),
                    "paths have been expanded and",
                    len(frontier),
                    "paths remain in the frontier",
                )
            return node
        explored.add(node.state)
        for idx, child in enumerate(node.expand(problem)):
            if child.state not in explored and child not in frontier:
                frontier.append(child, idx)
            elif child in frontier:
                if f(child) < frontier[child][0]:
                    del frontier[child]
                    frontier.append(child, idx)
    return None


def shxt_first_graph_search(problem, f, display=False):
    """Search the nodes with the shxtest f scores first."""
    f = memoize(f, "f")
    node = Node(problem.initial)
    frontier = deque()
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            if display:
                print(
                    len(explored),
                    "paths have been expanded and",
                    len(frontier),
                    "paths remain in the frontier",
                )
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[frontier.index(child)].path_cost:
                    del frontier[frontier.index(child)]
                    frontier.append(child)
    return None


def uniform_cost_search(problem, display=False):
    return shxt_first_graph_search(problem, lambda node: node.path_cost, display)


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n)."""
    h = memoize(h or problem.h, "h")
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


def astar_search_wrapper(problem, distance: str, display=False):
    if problem.matrix.shape == (10, 10) and distance == "manhattan":
        return uniform_cost_search(problem, display)
    elif distance == "euclidean":
        h = problem.euclidean_h
    elif distance == "manhattan":
        h = problem.manhattan_h
    else:
        raise ValueError()
    return astar_search(problem, h, display)


def get_matrix(matrix):
    strings = []
    for row in matrix:
        new_row = []
        for i in row:
            if np.isinf(i):
                new_row.append("X")
            elif np.isnan(i):
                new_row.append("*")
            else:
                new_row.append(str(int(i)))

        row_string = " ".join(new_row) + "\n"
        strings.append(row_string)
    return "".join(strings)


def get_state_solution(goal_node: Node, problem: Map):
    matrix = problem.matrix.copy()

    if goal_node:
        solution = goal_node.state_solutions()
        for step in solution:
            matrix[step] = np.nan
        return goal_node.path_cost, get_matrix(matrix)
    else:
        return None, "null"


def main(map_path: str, algorithm: str, heuristic: str = None):
    """
    [map_] specifies the path to map, which is a text file formatted according to this example (see next page):

    the map matrix should have spaces between characters

    10 10
    1 1
    10 10
    111111478X
    1111111588
    1111111467
    11111X1136
    11111X1111
    1111111111
    61111X1111
    771XXX1111
    8811111111
    X871111111

    The first line indicates the size of the map (rows by columns),
    while the second and third line represent the start and end positions respectively.
    The map data then follows, where all elevation values are integers from 0 to 9 inclusive.

    [algorithm] specifies the search algorithm to use, with the possible values of bfs, ucs, and astar.
    [heuristic] specifies the heuristic to use for A* search, with the possible values of euclidean and manhattan.
    This input is ignored for BFS and UCS.


    Your program must then print to standard output the path returned by the search algorithm, in the following format:

    ***111478X
    11*1111588
    11*******7
    11111X11*6
    11111X1**1
    1111111*11
    61111X1***
    771XXX111*
    881111111*
    X87111111*
    """

    with open(map_path, "rt") as file:
        map_ = file.readlines()

    start, end, matrix = parse_map_to_matrix(map_)
    map_instance = Map(start, end, matrix)

    if test:
        print(map_path)
        print(algorithm)
        print(heuristic)
        print(start, end)
        print(matrix)

    if algorithm == "bfs":
        goal_node = breadth_first_tree_search(map_instance)
        _, matrix = get_state_solution(goal_node, map_instance)
        return matrix

    elif algorithm == "ucs":
        goal_node = uniform_cost_search(map_instance)
        _, matrix = get_state_solution(goal_node, map_instance)
        return matrix

    elif algorithm == "astar":
        goal_node = astar_search_wrapper(map_instance, heuristic)
        _, matrix = get_state_solution(goal_node, map_instance)
        return matrix


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
