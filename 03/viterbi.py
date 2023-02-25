"""
Todo:
- reading matrix and parsing it
- viterbi forward algo
"""

import numpy as np


K_idxs = None
K = None
N = None
shapes = None


def read_txt(p_: str):
    global shapes
    with open(p_, "rt") as file:
        data = file.readlines()

        # parse shape
        shapes = data[0].split()
        shapes = [int(x) for x in shapes]

        # parse matrix
        matrix = data[1: 1 + shapes[0]]
        matrix = [[np.nan if x == "X" else 0 for x in x.split()] for x in matrix]
        matrix = np.array(matrix)

        # observes lines
        numbers_of_obs = int(data[shapes[0] + 1])

        # observes
        obs = data[shapes[0] + 2: shapes[0] + 2 + numbers_of_obs]
        obs = [[int(y) for y in list(x) if y.isdigit()] for x in obs]
        obs = np.array(obs)

        # pro
        prob = data[-1]
        prob = float(prob.strip("\n"))
        return shapes, matrix, numbers_of_obs, obs, prob


def find_neighbours(i, j, m: np.array):
    shape = m.shape
    NESW = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]
    encodings = [0 if 0 <= x[0] <= shape[0] - 1 and 0 <= x[1] <= shape[1] - 1 else 1 for x in NESW]
    for idx, digit in enumerate(encodings):
        if digit == 0 and m[NESW[idx]] != 0:
            encodings[idx] = 1
    return [NESW[i] for i, x in enumerate(encodings) if x == 0], encodings


def create_Tm(m: np.array):
    """K x K, K states"""
    global K_idxs, K
    # create Tm
    K_idxs = list(zip(*np.where(m == 0)))
    K = len(K_idxs)
    Tm = np.zeros((K, K))

    for Ki, (i, j) in enumerate(K_idxs):
        neighbours, _ = find_neighbours(i, j, m)
        for n in neighbours:
            Kj = K_idxs.index(n)
            Tm[Ki, Kj] = 1 / len(neighbours)
    return Tm


def error(o: list, r: list, eps: float):
    di = (np.array(o) != np.array(r)).sum()
    return (1 - eps) ** (4 - di) * eps ** di


def create_Em(observations: list, m: np.array, eps: float):
    """K x N observations, State vs """
    global K, N, K_idxs
    N = len(observations)
    Em = np.zeros((K, N))
    for Ki, i_j in enumerate(K_idxs):
        for oi, o in enumerate(observations):
            _, r = find_neighbours(*i_j, m)
            Em[Ki][oi] = error(o, r, eps)
    return Em


def viterbi_forward_algo(Tm, Em):
    """K x N """
    global K, N
    trellis = np.zeros((K, N))
    pi_s = np.array([1 / K] * K)

    trellis[:, 0] = pi_s * Em[:, 0]

    for j in range(1, N):
        for i in range(K):
            max_ = -np.inf
            for k in range(K):
                if k != i:
                    tmp_ = trellis[k, j - 1] * Tm[k, i] * Em[i, j]
                    if tmp_ > max_:
                        max_ = tmp_
            trellis[i, j] = max_
    return trellis


def viterbi_forward_algo2(Tm, Em):
    """K x N """
    global K, N
    trellis = np.zeros((K, N))
    pi_s = np.array([1 / K] * K)

    trellis[:, 0] = pi_s * Em[:, 0]

    for j in range(1, N):
        for i in range(K):
            p_vals = trellis[:, j - 1] * Tm[:, i] * Em[i, j]
            indices = (-p_vals).argsort()[:2]

            if indices[0] != i:
                trellis[i, j] = p_vals[indices[0]]
            else:
                trellis[i, j] = p_vals[indices[1]]

    return trellis


def main(*args, **kwargs):
    p = args[0]
    shapes, matrix, numbers_of_obs, obs, prob = read_txt(p)
    Tm = create_Tm(matrix)
    Em = create_Em(obs, matrix, eps=prob)
    trellis = viterbi_forward_algo2(Tm, Em)
    save_npz(trellis)
    return trellis


def save_npz(trellis):
    outputs = []
    for s in range(N):
        tmp_zeros = np.zeros(shapes)
        for i, val in enumerate(trellis[:, s]):
            tmp_zeros[K_idxs[i]] = val
        outputs.append(tmp_zeros.copy())
    np.savez("output.npz", *outputs)


if __name__ == "__main__":

    import sys
    input_args = sys.argv[1:]
    main(*input_args)


