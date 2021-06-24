import numpy as np
import pytest

from agents.agent_mcts.mcts import Tree, Node
from agents.common import initialize_game_state, apply_player_action, PlayerAction, BoardPiece
from agents.common import available_moves, opponent
from tests.test_common import prepare_board_for_testing, prepare_board_and_player_for_testing
from agents.agent_random.random import generate_move_random


def test_Tree():
    board = initialize_game_state()
    root = Node(board, BoardPiece(1))
    tree = Tree(root)
    assert isinstance(tree, Tree)
    assert tree.root == root
    with pytest.raises(TypeError):
        Tree(board)


def test_Tree_expand():
    board = initialize_game_state()
    root = Node(board, BoardPiece(1))
    tree = Tree(root)

    tree.expand(root)

    assert (tree.root.board == board).any()
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
    tree = Tree(root)
    return {"player": player, "board": board, "root": root, "tree": tree}


def test_Tree_expand_loop_root_empty():

    prep = initialize_random_root()
    for i in range(7):
        prep["tree"].expand(prep["root"])
    assert prep["root"].unexpanded == set()


def test_Tree_expand_too_much():
    prep = initialize_random_root()

    with pytest.raises(Exception):
        for i in range(8):
            prep["tree"].expand(prep["root"])


def test_Tree_playout():
    prep = initialize_random_root()




# def test_generate_move_mcts():
    """

    """
    # assert  # ...that the child denominators sum up to denominator
    # and child numerators sum up to the opposite
    # of it's parents nominator.
