import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, PlayerAction, GameState, lowest_free, check_end_state, SavedState


# Future, hypothetical boards
def level_search(
    board: np.ndarray, player: BoardPiece, state: Optional[SavedState]
    ) -> Tuple[PlayerAction, Optional[SavedState]]:
    depth = state[0]
    heuristic = state[1]
    hp_board = np.copy(board)
    for hp_action in np.ndarray([3, 4, 2, 5, 1, 6, 0]).astype(np.int8):
        # maydo: chose the column with gaussian-like distribution centered at three
        hp_board[lowest_free(hp_board, hp_action), hp_action] = player
        if check_end_state(hp_board, player) == GameState.IS_WIN:
            return (hp_action, heuristic+1)
        else:
            opponent = (player % 2) + 1
            if check_end_state(hp_board, opponent) == GameState.IS_WIN:
                heuristic = -257
            level_search(hp_board, opponent, (depth-1, heuristic/49))
    return None, None


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, state: Optional[SavedState]
    ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    For each action we consider all moves of opponent and choose
    - subtract ones that give an option of immediate win to the opponent, say "k" of them.
    - subtract ones that give a win option to the opponent in two moves, i.e.
    (1) 7 moves for me, 7 for the opponent

    (1) 7-k1 moves for me, 7 for the opponent,
    (2) 7 for me, and 7 for the opponent.

    (1) 7-k1 moves for me, 7 for the opponent,
    (2) for each 49 - 7k1 boards calculate
    7-k2 for me, and 7 for the opponent.
    Save the (2) part for the next round.
    for boards taken after two rounds make convolution only on the upper rows (use "lowest free" funciton)
    """
