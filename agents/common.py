import numpy as np
from enum import Enum
from typing import Optional, Callable, Tuple, TYPE_CHECKING

# side note: occasional problem with using the fftconvolve, check convolve2D
#if TYPE_CHECKING:
from scipy.signal import fftconvolve, convolve2d


class SavedState:
    pass


BoardPiece = np.int8  # The data type of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

BoardPiecePrint = str  # type for string representation of Board
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class GameDim(Enum):
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
    # Remark: since you've defined the parameters in an enumeration class (GameDimensions), why not use them here?
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
    # maydo: replace '0', '1',... with class player value
    mapping_inverse = np.array([('0', NO_PLAYER_PRINT), ('1', PLAYER1_PRINT), ('2', PLAYER2_PRINT)], str)
    mapping_inverse[:, [0, 1]] = mapping_inverse[:, [1, 0]]

    still_a_str = pp_board.replace('|', '')
    for (k, v) in mapping_inverse:
        still_a_str = still_a_str.replace(k, v)

    surged_str = still_a_str.split('\n')[1:7]  # check
    surged_string_array = np.zeros((6, 7), dtype=str)
    for i in range(GameDim.HEIGHT.value):
        row = list(surged_str[i])[::2]
        surged_string_array[i, :] = np.array(row)

    reboard = surged_string_array.astype(BoardPiece)
    reboard = np.flip(reboard, axis=0)
    return reboard


def lowest_free(board: np.ndarray, action: PlayerAction) -> int:
    """
    Given a board and an index of a column,
     it returns an integer, which is the index of the lowest row free to occupy in that column.
    """
    column = board[:, action]
    nonzero = np.nonzero(column)
    return int(nonzero[0].shape[0]) # Remark: maybe flattening nonzero first could save you from this rather cumbersome notation


def apply_player_action(
        board: np.ndarray, action: PlayerAction,
        player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row for the column with the number "action". The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy is True:
        copy_of_the_board = np.copy(board)
        copy_of_the_board[lowest_free(board, action), action] = player
        return copy_of_the_board
    else:
        i = lowest_free(board, action)
        board[i, action] = player
        return board


def connect(board: np.ndarray, player: BoardPiece, n=4) -> bool:
    """
    Using scipy convolve2d.
    :param board: array to check for connected pieces.
    :param player: the one that just moved, ie the board piece to check.
    :param n: numbers of pieces connected. Here it's four.
    :return: boolean: is there four connected pieces or not.
    """
    four = np.ones((1, n))
    four_connected = [four, four.T, np.diag(four.flatten()), np.fliplr(np.diag(four.reshape(4)))]

    # Convolution of kernel and the board will reveal connectedness.
    if player == BoardPiece(1):
        # It's important to exclude summing board pieces of nr two.
        board = np.where(board== 1, board, 0)

    for kernel in four_connected:
        convolution = convolve2d(board, kernel, "valid")
        is_connected = (convolution == player * n).any()
        if is_connected:
            return True

    return False



def check_end_state(
        board: np.ndarray, player: BoardPiece,
        last_action: Optional[PlayerAction] = None
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is the play on-going (GameState.STILL_PLAYING)?
    """
    # maydo: continue with TEST and design of this function.
    game_state = GameState.STILL_PLAYING

    if connect(board, player):
        game_state = GameState.IS_WIN
    elif (board != 0).all():
        game_state = GameState.IS_DRAW
    return game_state


def opponent(player: BoardPiece) -> BoardPiece:
    """
    Switches players. Call it just "opponent", to make it work with other funcitons.
    :param player:
    :return:
    """
    if player == 1:
        return BoardPiece(2)
    if player == 2:
        return BoardPiece(1)
    if player == 0:
        return player


def opponent_slower_version(player: BoardPiece) -> BoardPiece:
    op = BoardPiece(player%2 + 1)
    return op if player != 0 else BoardPiece(0)


def available_moves(board: np.ndarray) -> np.ndarray:
    """
    checks the moves still available of a board, returns a 1d vector of player action data type.
    """
    low_frees = np.zeros(board.shape[1])
    for j in range(board.shape[1]):
        low_frees[j] = lowest_free(board, j)
    return np.array(np.argwhere(low_frees < board.shape[0]).reshape(-1), dtype=PlayerAction)


def available_moves_another(board: np.ndarray) -> np.ndarray:
    # maydo: refactor, that the functions returns a set or a vector of seven booleans.
    # note: this one causes Type error, when calculating set of this return... Why?
    return np.array(np.argwhere(np.any(board == 0, axis=0)), dtype=PlayerAction)


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of generate_move function
]
