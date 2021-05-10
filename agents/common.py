import pdb

import numpy as np
from enum import Enum
from typing import Optional
from typing import Callable, Tuple
from scipy.signal import fftconvolve


class SavedState:
    pass


BoardPiece = np.int8  # The data type of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

BoardPiecePrint = str  # type for string representation of Board
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')  # nice representations \U+25FB  \U+25B2 ?
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class GameDimensions(Enum):
    HEIGHT = 6
    LENGTH = 7
    CONNECT = 4


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7)
    and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER)
    """
    board = np.zeros((6, 7), dtype=BoardPiece)
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    Human readable string representation of the board.
    To play and do diagnostics to the console (stdout).
    The piece board[0, 0] should appear in the lower-left.
    Just an example:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    # TODO: implement PLAYER1_PRINT and the other 2 variables.
    minus_board = np.flip(board, axis=0)
    pp_board = ''

    for i in np.arange(42):
        if i % 7 == 0:
            pp_board = pp_board + '|\n|'
        pp_board = pp_board + str(minus_board[i // 7, i % 7]) + ' '
    mapping_0 = [('0', NO_PLAYER_PRINT), ('1', PLAYER1_PRINT), ('2', PLAYER2_PRINT)]
    for k, v in mapping_0:
        pp_board = pp_board.replace(k, v)
    cardinals = '0 1 2 3 4 5 6 '
    pp_board = ("|" + 14*'=') + pp_board + ('|\n|' + 14*'=') + ('|\n|' + cardinals) + '|'
    return pp_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns back to ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    # maydo: replace '0', '1,... with class player value
    mapping_inverse = np.array([('0', NO_PLAYER_PRINT), ('1', PLAYER1_PRINT), ('2', PLAYER2_PRINT)], str)
    mapping_inverse[:, [0, 1]] = mapping_inverse[:, [1, 0]]

    still_a_str = pp_board.replace('|', '')
    for (k, v) in mapping_inverse:
        still_a_str = still_a_str.replace(k, v)

    surged_str = still_a_str.split('\n')[1:7]  # check
    surged_string_array = np.zeros((6, 7), dtype=str)
    print(surged_string_array.shape)
    for i in range(GameDimensions.HEIGHT.value):
        row = list(surged_str[i])[::2]
        surged_string_array[i, :] = np.array(row)

    reboard = initialize_game_state()
    reboard = surged_string_array.astype(BoardPiece)
    reboard = np.flip(reboard, axis=0)
    return reboard


def lowest_free(board: np.ndarray, action: PlayerAction) -> int:
    """
    Given a board and an index of a column,
     it returns the 2d index, the lowest row free to occupy in that column.
    """
    column = board[:, action]
    nonzero = np.nonzero(column)
    return int(nonzero[0].shape[0])


def apply_player_action(
        board: np.ndarray, action: PlayerAction,
        player: BoardPiece, copy: bool=False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy is True:
        copy_of_the_board = board.copy()
        copy_of_the_board[lowest_free(board, action), action] = player
        return copy_of_the_board
    else:
        board[np.nonzero(board[:, action])[0].shape[0], action] = player
    return board


def check_end_state(
        board: np.ndarray, player: BoardPiece,
        last_action: Optional[PlayerAction] = None
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """

    # winning kernels:
    four = np.ones((1, 4))
    four_connected = [four, four.T, np.diag(four.flatten()), np.fliplr(np.diag(four.reshape(4)))]

    game_state = GameState.STILL_PLAYING

    # convolution of four1 and the board (rectified) will reveal the win
    for four1 in four_connected:
        if player == PLAYER1:
            board_player1 = np.where(board == 1, board, 0)
            checking1 = np.round(fftconvolve(board_player1, four1, "valid"))
            win_Player1 = (checking1 == 4).any()
            if win_Player1:
                game_state = GameState.IS_WIN
        if player == PLAYER2:
            checking2 = np.round(fftconvolve(board, four1, "valid"))
            win_Player2 = (checking2 == 8).any()
            if win_Player2:
                game_state = GameState.IS_WIN
        if (board != 0).all() and game_state != GameState.IS_WIN:
            game_state = GameState.IS_DRAW
        else:
            if game_state != GameState.IS_WIN:
                game_state = GameState.STILL_PLAYING
    return game_state


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of generate_move function
]