from winequalityRFL import *
import numpy as np

train = "./data/train"
test = "./data/test"
test_res = "./data/test_res"

minleaf = 30


def read_res(p):
    data = open(p, "rt").read()
    return [int(x) for x in data.splitlines()]


def acc(y1, y2):
    y1_arr = np.array(y1)
    y2_arr = np.array(y2)
    return (y1_arr == y2_arr).sum() / len(y1)


if __name__ == "__main__":

    ys_t = read_res(test_res)
    ys = main(train, test, minleaf, n_trees=5, random_seed=41)

    print("acc:", acc(ys_t, ys))
