import numpy as np
import sys
import random

from collections import Counter


class Tree:
    def __init__(self, attr: int, split_val: float, attr_name: str = None):
        self.attr = attr
        self.attr_name = attr_name
        self.split_val = split_val
        self.left = None
        self.right = None

    def add(self, subtree, left=True):
        if left:
            self.left = subtree
        else:
            self.right = subtree

    def __call__(self, example):
        attr_val = example[self.attr]

        if attr_val > self.split_val:
            return self.right(example)
        else:
            return self.left(example)


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display(self):
        print("RESULT =", self.result)

    def __repr__(self):
        return repr(self.result)


def rows_same(data: np.array):
    arr = []
    for i in range(data.shape[1]):
        arr.append(np.unique(data[:, i]).size == 1)
    return all(arr)


def unique_mode(y):
    unique, counts = np.unique(y, return_counts=True)
    if (counts == max(counts)).sum() == 1:
        return unique[np.argmax(counts)]
    else:
        return "unknown"


def DTL(data, minleaf: int):

    N = len(data)

    if N <= minleaf or np.unique(data[:, -1]).size == 1 or rows_same(data):
        val = unique_mode(data[:, -1])
        return DecisionLeaf(val)

    attr, split_val = choose_split(data)

    node = Tree(attr, split_val)
    node.add(DTL(data[data[:, attr] <= split_val], minleaf=minleaf), left=True)
    node.add(DTL(data[data[:, attr] > split_val], minleaf=minleaf), left=False)
    return node


def i(data):
    unq, cnts = np.unique(data[:, -1], return_counts=True)
    total = cnts.sum()
    return sum(-1 * (c / total) * np.log2(c / total) for c in cnts)


def choose_split(data):

    n_attr = data.shape[-1] - 1
    n_examples = data.shape[0]

    best_gain = 0

    i_parent = i(data)

    for attr in range(n_attr):
        vals = np.sort(data[:, attr])
        split_vals = (vals[:-1] + vals[1:]) / 2

        for split_val in split_vals.tolist():

            left = data[data[:, attr] <= split_val]
            right = data[data[:, attr] > split_val]
            gain = i_parent - sum(
                [len(left) / n_examples * i(left), len(right) / n_examples * i(right)]
            )

            if gain > best_gain:
                best_gain = gain
                best_split_val = split_val
                best_attr = attr

    return best_attr, best_split_val


def open_data(p, mode="r"):
    return open(p, mode=mode).read()


def num_or_str(x):
    """The argument is a string; convert to a number if possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def parse_csv(input_p, delim=" "):
    lines = [line for line in input_p.splitlines() if line.strip()]
    data = [
        list(map(num_or_str, [x for x in line.split(delim) if x])) for line in lines
    ]
    headers, data = data[0], data[1:]
    return headers, data


def RFL(data, minleaf, n_trees):

    forkest = []
    N = len(data)
    sample_indexes = [random.randint(0, N - 1) for _ in range(n_trees * N)]

    for idx in range(n_trees):
        sampled_data = data[sample_indexes[idx * N : idx * N + N]]
        n = DTL(sampled_data, minleaf)
        forkest.append(n)

    return forkest


def predict_rfl(forest, data):

    yss = []

    for e in data:
        ys = []
        for tree in forest:
            ys.append(tree(e))

        b = Counter(ys)
        y = b.most_common(1)[0][0]
        yss.append(y)

    return yss


def main(train, test, minleaf, n_trees, random_seed):

    if isinstance(minleaf, str):
        minleaf = int(minleaf)

    if isinstance(n_trees, str):
        n_trees = int(n_trees)

    if isinstance(random_seed, str):
        random_seed = int(random_seed)

    random.seed(random_seed)

    headers_train, data_train = parse_csv(open_data(train))
    headers_test, data_test = parse_csv(open_data(test))

    train_arrays = np.array(data_train)
    test_arrays = np.array(data_test)

    trees = RFL(train_arrays, minleaf=minleaf, n_trees=n_trees)

    ys = predict_rfl(trees, test_arrays)

    for y in ys:
        print(int(y))

    return ys


if __name__ == "__main__":
    main(*sys.argv[1:])
