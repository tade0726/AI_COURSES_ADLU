from pathfinder import *


def test_bfs():
    print("bfs:")
    start, end, matrix = parse_map_to_matrix(amap)
    map_i = Map(start, end, matrix)
    goal_node = breadth_first_tree_search(map_i)
    print(goal_node.state_solutions())
    print()
    print(get_matrix(matrix))
    print()
    cost, matrix = get_state_solution(goal_node, map_i)
    print("cost:", cost)
    print(matrix)
    print()


def test_ucs():
    print("ucs: ")
    start, end, matrix = parse_map_to_matrix(amap)
    map_i = Map(start, end, matrix)
    goal_node = uniform_cost_search(map_i, True)
    print(goal_node.state_solutions())
    print()
    print(get_matrix(matrix))
    print()
    cost, matrix = get_state_solution(goal_node, map_i)
    print("cost:", cost)
    print(matrix)
    print()


def test_a_star_euc():
    # a star -- euclidean
    print("a star euclidean: ")
    start, end, matrix = parse_map_to_matrix(amap)
    map_i = Map(start, end, matrix)
    goal_node = astar_search_wrapper(map_i, "euclidean", True)
    print(goal_node.state_solutions())
    print()
    print(get_matrix(matrix))
    print()
    cost, matrix = get_state_solution(goal_node, map_i)
    print("cost:", cost)
    print(matrix)
    print()


def test_a_star_manh():
    # a star -- manhattan
    print("a star manhattan: ")
    start, end, matrix = parse_map_to_matrix(amap)
    map_i = Map(start, end, matrix)
    goal_node = astar_search_wrapper(map_i, "manhattan", True)
    print(goal_node.state_solutions())
    print()
    print(get_matrix(matrix))
    print()
    cost, matrix = get_state_solution(goal_node, map_i)
    print("cost:", cost)
    print(matrix)
    print()


if __name__ == "__main__":

    amap = "test1.txt"
    print(amap)
    print()
    algo = "bfs"
    heur = None

    argv = (amap, algo, heur)
    print(main(*argv))

    amap = "test2.txt"
    print(amap)
    print()
    algo = "ucs"
    heur = None

    argv = (amap, algo, heur)
    print(main(*argv))

    amap = "test3.txt"
    print(amap)
    print()
    algo = "astar"
    heur = "manhattan"

    argv = (amap, algo, heur)
    print(main(*argv))

    amap = "test4.txt"
    print(amap)
    print()
    algo = "astar"
    heur = "euclidean"

    argv = (amap, algo, heur)
    print(main(*argv))

    amap = "test5.txt"
    print(amap)
    print()
    algo = "astar"
    heur = "manhattan"

    argv = (amap, algo, heur)
    print(main(*argv))

    amap = "test6.txt"
    print(amap)
    print()
    algo = "astar"
    heur = "euclidean"

    argv = (amap, algo, heur)
    print(main(*argv))
