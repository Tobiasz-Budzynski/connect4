import numpy as np
import timeit


def test_generate_move():
    """
    it's name is generate move random in the file random,
    while it is imported as generate move from initial file
    :return:
    """
    from agents.agent_random import generate_move
    from agents.common import GenMove, apply_player_action, BoardPiece
    from tests.test_common import prepare_board_for_testing
    # maydo: check if the moves are legal

    player = np.random.choice([1, 2]).astype(BoardPiece)
    board, low_frees = prepare_board_for_testing()
    action = generate_move(board, player, None)[0]
    print(timeit.timeit(stmt="while n<1000: x=x+'A'; n+=1", setup='n=0; x=""', number=1000))
    board = apply_player_action(board, action, player)
    assert type(generate_move) == GenMove
    assert isinstance(generate_move, GenMove)

