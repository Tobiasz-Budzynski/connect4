import numpy as np
from enum import Enum
from agents.common import BoardPiece, NO_PLAYER, PlayerAction, GameState
import pdb
rng = np.random.default_rng()


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

# maydo: From a game state make a data type.


def test_pretty_print_board():
    from agents.common import pretty_print_board
    from agents.common import initialize_game_state

    rety = initialize_game_state()
    boardy = pretty_print_board(rety)

    assert isinstance(boardy, str)
    assert len(boardy.splitlines()) == 9


def test_string_to_board():
    from agents.common import pretty_print_board
    from agents.common import initialize_game_state
    from agents.common import string_to_board

    '''
    cut beginning and the end
    cut boarders
    split when new line (slash n)
    pick the part without space (every second),
    by taking even indexes
    '''

    example_game_state = initialize_game_state()
    example_game_state[0,1] = BoardPiece(2)
    boardy = pretty_print_board(example_game_state)
    rerety = string_to_board(boardy)

    assert isinstance(rerety, np.ndarray)
    assert rerety.dtype == BoardPiece
    assert rerety.shape == (6, 7)
    assert (rerety == example_game_state).all()
# maydo: Tail recurrsive search in the string.


def prepare_board_for_testing(full=False):
    from agents.common import initialize_game_state

    print('') #for better view when testing
    board0 = initialize_game_state()
    if full == True:
        low_frees = 6*np.ones((7,))
    else:
        low_frees = rng.integers(low=0, high=7, size=7)

    for i in range(6):
        for j in range(7):
            if i < low_frees[j]:
                board0[i, j] = BoardPiece(rng.integers(low=1, high=3))
    return board0, low_frees


def test_prepare_board_for_testing():
    board, low_frees = prepare_board_for_testing(full=False)
    assert board.shape == (6, 7)


def test_lowest_free():
    from agents.common import lowest_free, initialize_game_state

    board1 = initialize_game_state()
    board1[0,3] = BoardPiece(1)
    column1 = np.int8(3)
    index = np.array([lowest_free(board1, column1), column1])
    for i in (0, 1):
        assert 0 <= index[i] < board1.shape[i]
    assert board1[index[0], index[1]] == NO_PLAYER


def test_apply_player_action():
    from agents.common import apply_player_action
    from agents.common import pretty_print_board

    board0, low_frees = prepare_board_for_testing()

    change = np.zeros((6, 7))
    action = rng.integers(low=0, high=6)
    player = BoardPiece(rng.integers(low=1, high=3))
    change[low_frees[action], action] = player

    post_board = apply_player_action(board0, action, player, copy=True)
    print(pretty_print_board(board0))
    print(pretty_print_board(post_board))
    assert isinstance(post_board, np.ndarray)
    assert post_board.shape == (6, 7)
    assert (post_board == board0 + change).all()


def test_connect():
    from agents.common import connect, pretty_print_board, PLAYER1_PRINT, PLAYER2_PRINT
    for i in range(10):
        board, low_free = prepare_board_for_testing(full=True)
        player = BoardPiece(np.random.randint(1,3))
        print(pretty_print_board(board))
        connect4 = connect(board, player)
        if player == 1: symbol = PLAYER1_PRINT
        else: symbol = PLAYER2_PRINT
        print("Connected 4 ", symbol," is ", connect4)
    rate = np.zeros((4,))
    for i in range(1000):
        board, low_free = prepare_board_for_testing(full=True)
        wunth = connect(board, BoardPiece(1))
        twoth = connect(board, BoardPiece(2))
        if wunth and twoth: rate[-1] += 1
        elif wunth: rate[1] += 1
        elif twoth: rate[2] += 1
        else: rate[0] += 1
    rate *= 1 / 1000
    rate = np.round(rate, 4)
    print("rate_draw, rate_1, rate_2, rate_1_2 are ", *rate)
    assert type(connect4) == np.bool_


def test_check_end_state():
    from agents.common import check_end_state

    for i in range(1017):
        board, low_frees = prepare_board_for_testing()
        player = BoardPiece(rng.integers(low=1, high=3))
  # maydo: implementing last action parameter
        now_game = check_end_state(board, player)
        assert isinstance(now_game, Enum)
        assert now_game in GameState.__dict__.values()
