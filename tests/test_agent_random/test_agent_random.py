import numpy as np

def test_generate_move():
    """
    it's name is generate move random in the file random,
    while it is imported as generate move from initial file
    :return:
    """
    from agents.agent_random import generate_move
    from agents.common import GenMove, lowest_free, apply_player_action, BoardPiece
    from tests.test_common import prepare_board_for_testing
    # todo: check if the moves are legal
    player = np.random.choice([1, 2]).astype(BoardPiece)
    board, low_frees = prepare_board_for_testing()
    action = generate_move(board, player)
    board = apply_player_action(board, action, player)
    assert type(generate_move()) == GenMove

