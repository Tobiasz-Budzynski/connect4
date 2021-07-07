import numpy as np
from enum import Enum
from agents.common import BoardPiece, NO_PLAYER, PlayerAction, GameState
from agents.common import initialize_game_state, apply_player_action, opponent

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
    from agents.common import pretty_print_board, initialize_game_state
    from agents.common import NO_PLAYER_PRINT, PLAYER1_PRINT, PLAYER2_PRINT

    board_tppb = initialize_game_state()
    allowed = [
        '=', '|', ' ', '\n',
        '0', '1', '2', '3', '4', '5', '6', NO_PLAYER_PRINT, PLAYER1_PRINT, PLAYER2_PRINT
    ]
    for i in range(6 * 7):
        p_print_b = pretty_print_board(board_tppb)
        piece = BoardPiece(np.int8(np.random.randint(3)))
        board_tppb[i // 7, i % 7] = piece
        assert isinstance(p_print_b, str)
        assert len(p_print_b.splitlines()) == 9
        assert len(p_print_b) == 152
        for ch in p_print_b:
            assert ch in allowed
    print(board_tppb)
    print(p_print_b)


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
    example_game_state[0, 1] = BoardPiece(2)
    boardy = pretty_print_board(example_game_state)
    rerety = string_to_board(boardy)

    assert isinstance(rerety, np.ndarray)
    assert rerety.dtype == BoardPiece
    assert rerety.shape == (6, 7)
    assert (rerety == example_game_state).all()


# maydo: Tail recursive search in the string.
# Remark: you're not testing if the function works specifically. You should test at least one specific example. You're
#         basically only testing, whether pretty_print_board and string_to_board are inverses of each other.
#         Also, using other functions in tests is dangerous, because it can introduce circular dependencies.


def prepare_board_and_player_for_testing():
    # todo: choose first an int from range(6*7)

    from agents.agent_random.random import generate_move_random
    from agents.common import connect

    while True:
        probabilities = np.flip(np.array(range(6*7-2))+20)/sum(np.flip(np.array(range(6*7-2))+20))
        length = np.random.choice(1+np.array(range(6*7-2)), p=probabilities)
        board = initialize_game_state()
        player = BoardPiece(np.random.randint(1, 3))

        for i in range(length):
            move = generate_move_random(board, player, None)
            board = apply_player_action(board, move, player)
            player = opponent(player)
        if not (connect(board, player) or connect(board, opponent(player
                                                                  ))):
            break
    return board, player


def test_prepare_board_and_player_for_testing():
    # from agents.common import pretty_print_board
    board, player = prepare_board_and_player_for_testing()
    assert isinstance(board, np.ndarray)


def prepare_board_for_testing(full=False):
    #
    print('')  # for better view when testing
    board0 = initialize_game_state()
    if full == True:
        low_frees = 6 * np.ones((7,))
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
    # Remark: again too unspecific


def test_lowest_free():
    from agents.common import lowest_free, initialize_game_state

    board1 = initialize_game_state()
    board1[0, 3] = BoardPiece(1)
    column1 = np.int8(3)
    index = np.array([lowest_free(board1, column1), column1])
    for i in (0, 1):
        assert 0 <= index[i] < board1.shape[i]
    # Remark: you should specifically test if the lowest free row you find is correct. This test is too complicated, which
    #         obscures the fact that it's not very useful
    assert board1[index[0], index[1]] == NO_PLAYER  # Remark: this has nothing to do with the function


def test_apply_player_action():
    from agents.common import apply_player_action
    from agents.common import pretty_print_board

    board0, low_frees = prepare_board_for_testing()
    change = np.zeros((6, 7))
    action = rng.integers(low=0, high=6)
    player = BoardPiece(rng.integers(low=1, high=3))
    change[low_frees[action], action] = player

    post_board = apply_player_action(board0, action, player, copy=True)

    print(pretty_print_board(post_board))
    assert isinstance(post_board, np.ndarray)
    assert post_board.shape == (6, 7)
    assert (post_board == board0 + change).all()


def test_connect():
    from agents.common import connect, PLAYER1_PRINT, PLAYER2_PRINT

    # Visual checking for correctness of the "connect" function. Uncomment the prints to use.
    for i in range(5):
        board, low_free = prepare_board_for_testing(full=True)
        player = BoardPiece(np.random.randint(1, 3))
        # print(pretty_print_board(board))
        connected = connect(board, player)
        if player == 1:
            symbol = PLAYER1_PRINT
        else:
            symbol = PLAYER2_PRINT
        # print("Connected 4 ", symbol, " is ", connect4)

    four = np.ones((1, 4), dtype=BoardPiece)
    four_connected = [four, four.T, np.diag(four.flatten()), np.fliplr(np.diag(four.reshape(4)))]
    board_zeros = np.full_like(board, 0)

    for kernel in four_connected:
        board[:kernel.shape[0], :kernel.shape[1]] = kernel
        board_kernel = board_zeros.copy()
        board_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel

    assert isinstance(connected, bool)
    assert True is connect(board, BoardPiece(1))
    assert True is connect(board_kernel, BoardPiece(1))
    assert False is connect(board_kernel, BoardPiece(2))
    assert False is connect(board_zeros, BoardPiece(1))
    assert False is connect(board_zeros, BoardPiece(2))

    # Calculate statistics of random full board.
    # Draws, 1st Wins, 2nd Wins, both Wins.
    rate = np.zeros((4,))
    for i in range(1000):
        board, low_free = prepare_board_for_testing(full=True)
        wunth = connect(board, BoardPiece(1))
        twoth = connect(board, BoardPiece(2))
        if wunth and twoth:
            rate[-1] += 1
        elif wunth:
            rate[1] += 1
        elif twoth:
            rate[2] += 1
        else:
            rate[0] += 1
    rate *= 1 / 1000
    rate = np.round(rate, 4)
    print("rate_draw, rate_1, rate_2, rate_1_2 are ", *rate)


def test_check_end_state():
    from agents.common import check_end_state, pretty_print_board
    for i in range(1000):
        board, low_frees = prepare_board_for_testing()
        player = BoardPiece(rng.integers(low=1, high=3))
        if i < 3:
            board[0][0:5] = BoardPiece(2)
            now_game = check_end_state(board, player)
            pretty_print_board(board)
            print(now_game)
        # maydo: implementing last action parameter
        now_game = check_end_state(board, player)
        assert isinstance(now_game, Enum)
        assert now_game in GameState.__dict__.values()
        # Remark: This test is unspecific again, you're only checking whether you return a GameState instance.
        #         PyCharm more or less does that for you due to the type hints.


def test_opponent():
    from agents.common import opponent

    assert opponent(0) == BoardPiece(0)
    assert opponent(BoardPiece(1)) == BoardPiece(2)
    assert (type(opponent(BoardPiece(i))) == BoardPiece for i in [0, 1, 2])
    assert opponent(BoardPiece(2)) == BoardPiece(1)


def test_available_moves():
    pass
    from agents.common import available_moves

def test_mcts_passing_save_state():
    from main import human_vs_agent
    from agents.agent_mcts.mcts import generate_move_mcts
    from agents.agent_random.random import generate_move_random
    human_vs_agent(generate_move_mcts, generate_move_random)
