import copy
import numpy as np
import pytest
import cProfile

from agents.agent_mcts.mcts import MCTS, Node, generate_move_mcts
from agents.common import initialize_game_state, apply_player_action,\
    PlayerAction, BoardPiece, GameState, available_moves, opponent, check_end_state
from tests.test_common import prepare_board_for_testing, prepare_board_and_player_for_testing
from agents.agent_random.random import generate_move_random


def test_MCST():
    board = initialize_game_state()
    root = Node(board, BoardPiece(1))
    tree = MCTS(root)
    assert isinstance(tree, MCTS)
    assert tree.root == root
    with pytest.raises(TypeError):
        MCTS(board)


def test_MCTS_expand():
    board = initialize_game_state()
    root = Node(board, BoardPiece(1))
    tree = MCTS(root)

    tree.expand(root)

    assert (tree.root.board == board).all()
    action, child_node = list(root.children.items())[0]
    assert action in range(7)
    assert isinstance(child_node, Node)
    assert child_node.parent == root
    new_unexpanded = set(range(7))
    new_unexpanded.remove(action)
    assert root.unexpanded == new_unexpanded


def initialize_random_root():
    board, player = prepare_board_and_player_for_testing()

    root = Node(board, player)
    tree = MCTS(root)
    return {"player": player, "board": board, "root": root, "tree": tree}


def test_MCTS_expand_loop_root_empty():

    prep = initialize_random_root()
    for i in range(len(available_moves(prep["root"].board))):
        prep["tree"].expand(prep["root"])
    assert prep["root"].unexpanded == set()


def test_MCTS_expand_too_much():
    prep = initialize_random_root()

    with pytest.raises(Exception):
        for i in range(8):
            prep["tree"].expand(prep["root"])


def test_MCTS_playout_end_node_logic():
    situation = initialize_random_root()

    # put opponent pieces to the first column in the node's board.
    # maydo: randomize this test
    situation["root"].board[:, 0] = opponent(situation["player"])
    result = situation["tree"].playout(situation["root"])
    assert situation["root"]._count == 0
    assert isinstance(result[0], Node)
    assert isinstance(result[1], GameState)
    assert result[1] != GameState.STILL_PLAYING
    assert isinstance(result[2], BoardPiece)


def test_Tree_playout_proper():

    situation = initialize_random_root()
    situation["tree"].expand(situation["root"])
    result = situation["tree"].playout(situation["root"])
    print("\n \n Game's state and a nr of the player after playout are: ", result)
    assert isinstance(result[0], Node)
    assert isinstance(result[1], GameState)
    assert result[1] != GameState.STILL_PLAYING
    assert isinstance(result[2], BoardPiece)
    assert situation["root"]._count in range(6*7+1)


def test_tree_backprop_first_node_and_stats():
    """
    Check if the trials are increased and sometimes wins, according to winner.
    """
    import copy
    trials = 100
    stats = [0, 0, 0]

    for trial in range(trials):

        prep = initialize_random_root()
        child = MCTS.expand(prep["root"])
        child_frozen = copy.deepcopy(child)
        node, state, last_player = MCTS.playout(child)

        prep["tree"].backprop(child, state, last_player)

        assert child_frozen.trials +1 == child.trials
        if last_player == child.player and state == GameState.IS_WIN:
            assert child.wins == 0
            stats[0] += 1
        elif last_player == opponent(child.player) and state == GameState.IS_WIN:
            assert child.wins == 1
            assert child.wins == child_frozen.wins +1
            stats[1] += 1
        elif state == GameState.IS_DRAW:
            stats[2] += 1

    print("\n \n How many times the child won, root won, was a draw?", stats, " out of ", trials)


def prep_random_tree():
    """
    Here tree is most probably with some branches from the root only.
    :return:
    """
    prep = initialize_random_root()
    node = prep["root"]
    for i in range(np.random.randint(len(node.unexpanded))):
        places_left_in_board = np.abs(6*7 - np.count_nonzero(node.board) - 1)
        if places_left_in_board != 0:
            for count_expand in range(np.minimum(10, np.random.randint(places_left_in_board))):
                # maydo: rewrite.
                # To avoid calling for expansion a node with empty unexpanded set - the <<if>> clause is used:
                if node.unexpanded != set() and check_end_state(node.board, opponent(node.player)) == GameState.STILL_PLAYING:
                    node = MCTS.expand(node)
    return prep, node


def test_backprop_trials():
    count = 0
    prep, leaf = prep_random_tree()
    prep_old = copy.deepcopy(prep)
    leaf_old = copy.deepcopy(leaf)
    print("old root trials", prep_old["root"].trials)

    for state in GameState:
        for last_player in [1, 2]:

            prep["tree"].backprop(leaf, state, last_player)

            count += 1

            assert prep_old["root"].trials +count == prep["root"].trials
            assert leaf_old.trials +count == leaf.trials


def test_backprop_trials():
    # maydo: (should do) test deeper than one parent of a leaf
    for state in GameState:
        for last_player in [1, 2]:
            prep, leaf = prep_random_tree()
            prep_old = copy.deepcopy(prep)
            leaf_old = copy.deepcopy(leaf)
            prep["tree"].backprop(leaf, state, last_player)

            if state == GameState.IS_WIN and leaf.parent != "root":
                if last_player == leaf.player:
                    assert leaf.parent.wins == (leaf_old.parent.wins +1)
                    assert leaf.wins == leaf_old.wins
                if last_player == opponent(leaf.player):
                    assert leaf.parent.wins == leaf_old.parent.wins
                    assert leaf.wins == (leaf_old.wins +1)


def test_MCTS_select_next_child():
    # To change probablilities of initial board - look at probabilities variable in test common, prepare board and player.

    while True:

        prep, leaf = prep_random_tree()
        tree = prep["tree"]
        root = prep["root"]
        if check_end_state(leaf.board, leaf.player) == GameState.STILL_PLAYING:

            child = tree.expand(leaf)
            if check_end_state(child.board, child.player) == GameState.STILL_PLAYING:
                break

    selected_child = tree._select_next_child(root)

    while selected_child.children != {}:
        selected_child = tree._select_next_child(selected_child)

    simulation = MCTS.playout(child)
    tree.backprop(*simulation)


def test_loop_of_MCTS_methods():
    """
    Select.
    Expand.
    Playout.
    Backpropagation.
    Start from a root (no children). Select returns the root. Than expand.
    Than backprogation. Make it in a loop with a variable length. Finish.
    Check for the best action from the root.
    Play it against the random player.
    :return:
    """
    board, player = prepare_board_and_player_for_testing()
    root = Node(board, player)
    tree = MCTS(root)

    leaf = tree.select()
    new_leaf = tree.expand(leaf)
    last_leaf, state, last_player = tree.playout(new_leaf)
    tree.backprop(last_leaf, state, last_player)

def test_loop_of_MCTS_methods_abrev():

    root = Node(*prepare_board_and_player_for_testing())
    t = MCTS(root)  # stands for a tree

    for loop in range(1000):
        t.backprop(*t.playout(t.expand(t.select())))

    assert root.unexpanded == set()
    if root.children != dict():
        child = list(root.children.values())[0]
        assert child.unexpanded == set() or available_moves(child.board) == []

    print(t.root)
    # todo: test it against random : ]


def test_generate_move_mcts_avoid_immediate_loss():
    for loop in range(1):
        board = initialize_game_state()
        board[0, 2:5] = BoardPiece(1)
        board[0, 0:2] = BoardPiece(2)

        # cProfile.runctx('g(x,t,s)', {'g': generate_move_mcts, 'x': board, 't': BoardPiece(2), 's': None}, {})
        move, saved_state = generate_move_mcts(board, BoardPiece(2), None)
        assert move == 5

        print(board)


def test_generate_move_mcts_vs_random():
    """

    """

    # assert  # ...that the child denominators sum up to denominator
    # and child numerators sum up to the opposite
    # of it's parents nominator.








