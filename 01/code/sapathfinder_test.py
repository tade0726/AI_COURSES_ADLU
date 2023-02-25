from sapathfinder import *


if __name__ == "__main__":

    map_ = "test7.txt"
    init = "test7_init_path.txt"
    tini = 10
    tfin = 0.001
    alpha = 0.99
    d = 5

    argv = (map_, init, tini, tfin, alpha, d)
    main(*argv)
