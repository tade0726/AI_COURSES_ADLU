from collections import deque
import numpy as np
import random
import math


test = False

### --- utils ---


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


### --- utils ---


### --- meta class ---
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

    @staticmethod
    def path_cost(c, state1, action, state2):
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

    def expand(self, problem, random_actions=False):
        """List the nodes reachable in one step from this node."""
        return [
            self.child_node(problem, action)
            for action in problem.actions(self.state, random_actions)
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


### --- meta class ---


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

    def actions(self, state, random_actions=False):
        m, n = state
        next_actions = [
            (m - 1, n),
            (m + 1, n),
            (m, n - 1),
            (m, n + 1),
        ]  # up, down, left. right
        # random actions for annealing only
        if random_actions:
            random.shuffle(next_actions)
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


def breadth_first_tree_search(problem, random_actions=False, explored: set = None):
    """
    Search the shallowest nodes in the search tree first.
    """

    frontier = deque([Node(problem.initial)])  # FIFO queue
    explored = set() if explored is None else explored

    while frontier:
        node = frontier.popleft()

        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem, random_actions):
            if child.state not in explored:
                frontier.append(child)
    return None


def float_list(numbers_in_str):
    numbers = []
    for i in numbers_in_str:
        if not i:
            continue

        if i == "X":
            i_ = np.inf
        elif i == "*":
            i_ = np.nan
        else:
            i_ = float(i)
        numbers.append(i_)
    return numbers


def int_list(numbers_in_str):
    return [int(i) for i in numbers_in_str]


def parse_map_to_matrix(lines: str):
    m = []
    lines = [line.strip("\n") for line in lines]

    start = int_list(lines[1].split(" "))
    end = int_list(lines[2].split(" "))

    start = (start[0] - 1, start[1] - 1)
    end = (end[0] - 1, end[1] - 1)

    for line in lines[3:]:
        if not line:
            continue
        ns = float_list(line.split(" "))
        m.append(ns)

    return start, end, np.array(m)


def read_map(path: str):
    """return matrix"""
    with open(path, "rt") as file:
        map_ = file.readlines()

    start, end, matrix = parse_map_to_matrix(map_)
    return start, end, matrix


def read_init(path: str):
    with open(path, "rt") as file:
        lines = file.readlines()

    lines = [line.strip("\n") for line in lines]

    m = []
    for line in lines:
        if not line:
            continue
        ns = float_list(line.split(" "))
        m.append(ns)
    return np.array(m)


def path_parse(start, end, path_matrix):
    """return the init path"""
    rows, columns = np.where(np.isnan(path_matrix))
    position = list(zip(rows, columns))
    new_matrix = np.ones_like(path_matrix) * np.inf

    for pos in position:
        new_matrix[pos] = 1

    map_instance = Map(start, end, new_matrix)
    goal_node = breadth_first_tree_search(map_instance)
    return goal_node.state_solutions()


def rand_local_adjust(path: list, d: int, matrix: np.array):
    s = random.choice(path)
    s_index = path.index(s)

    adjustment_path = path[s_index : s_index + d].copy()
    e = adjustment_path[-1]
    e_index = path.index(e)

    map_instance = Map(s, e, matrix)
    goal_node = breadth_first_tree_search(
        map_instance,
        explored=set(path[:s_index]) & set(path[e_index + 1 :]),
        random_actions=True,
    )
    path[s_index : e_index + 1] = goal_node.state_solutions()
    return path


def g_cost(path: list, matrix: np.array):
    cost = 0

    xp, yp = None, None

    for i, (x, y) in enumerate(path):
        if i > 0:
            extra_cost = matrix[(x, y)] - matrix[(xp, yp)]
            extra_cost = extra_cost if extra_cost > 0 else 0
            cost = cost + 1 + extra_cost
        xp, yp = x, y
    return cost


def prob_update(delta_g: int, T: float):
    return math.pow(math.e, delta_g / T)


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
    return "".join(strings).rstrip("\n")


def main(map_, init, tini, tfin, alpha, d):

    # parameters to numeric

    tini = float(tini)
    tfin = float(tfin)
    alpha = float(alpha)
    d = int(d)

    start, end, matrix = read_map(map_)
    path_init_matrix = read_init(init)

    if test:
        print(map_, init, tini, tfin, alpha, d)
        print(path_init_matrix)
        return

    init_path: list = path_parse(start, end, path_init_matrix)

    T = tini

    p = init_path
    p_cost = None

    records = []

    while T > tfin:

        p_new = rand_local_adjust(p, d, matrix)

        pn_cost = g_cost(p_new, matrix)
        if p_cost is None:
            p_cost = g_cost(p, matrix)

        delta_g = p_cost - pn_cost

        if delta_g > 0:
            p = p_new
            p_cost = pn_cost
        else:
            if np.random.binomial(1, prob_update(delta_g, T)):
                p = p_new
                p_cost = pn_cost

        records.append((T, p_cost))
        T = alpha * T

    for step in p:
        matrix[step] = np.nan
    print(get_matrix(matrix))

    for (t, c) in records:
        print(f"T = {t:06f}, cost = {c}")

    return p, p_cost


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
