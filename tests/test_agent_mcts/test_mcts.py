import copy

import numpy as np
import pytest

from agents.agent_mcts.mcts import MCTS, Node
from agents.common import initialize_game_state, apply_player_action,\
    PlayerAction, BoardPiece, GameState, available_moves, opponent
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
    assert isinstance(result[0], GameState)
    assert result[0] != GameState.STILL_PLAYING
    assert isinstance(result[1], BoardPiece)


def test_Tree_playout_proper():

    situation = initialize_random_root()
    situation["tree"].expand(situation["root"])
    result = situation["tree"].playout(situation["root"])
    print("\n \n Game's state and a nr of the player after playout are: ", result)
    assert isinstance(result[0], GameState)
    assert result[0] != GameState.STILL_PLAYING
    assert isinstance(result[1], BoardPiece)
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
        state, last_player = MCTS.playout(child)

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
            for count_expand in range(np.maximum(10, np.random.randint(places_left_in_board))):
                # side note: Every few trials a bug in design made it
                # calling for expansion a node with empty unexpanded set. So, I use the <<if>> clause.:
                if node.unexpanded != set():
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


def test_MCTS_select():




# def test_generate_move_mcts():
    """

    """
    # assert  # ...that the child denominators sum up to denominator
    # and child numerators sum up to the opposite
    # of it's parents nominator.








