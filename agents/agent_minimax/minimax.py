import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, PlayerAction, GameState, SavedState, lowest_free, check_end_state, connect

class SavedMM(SavedState):
    def __init__(self, depth_1, heuristic_1, story_1):
        self.depth = depth_1
        self.heuristic = heuristic_1
        self.story = story_1
# Future, hypothetical boards

def level_search(
    board: np.ndarray, player: BoardPiece, state: Optional[SavedState]
    ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    The function checks if action leads to win, if Yes it returns it,
    if No it avoids opponents wins. For each opponent response a heuristic is calculated.
    It increments, when there is a win opportunity for the player.
    At the end of the function heuristic is normalized according to depth,
    to have values from 0 to 1 (sth like a likelyhood of winning),
    unless heuristic is -infty, which means that opponent can force a win.
    """
    depth = state.depth
    heuristic = state.heuristic
    story = state.story
    heuristic_norm = 7**depth
    opponent = (player % 2) + 1
    hp_board = np.copy(board)  # abbreviation of hypothetical
    for hp_action in np.ndarray([3, 4, 2, 5, 1, 6, 0]).astype(np.int8):
        # maydo: chose the column with an order deterministic pseudo-gaussian-like centered at three.
        # ...and change the order after each game.
        hp_board[lowest_free(hp_board, hp_action), hp_action] = player
        if connect(hp_board, player, n=4):
            if depth == state[0]:
                return hp_action, heuristic+7**depth
            else:
                heuristic += 7**depth
        else:
            for hp_reaction in np.ndarray([3, 4, 2, 5, 1, 6, 0]).astype(np.int8):
                hp_board[lowest_free(hp_board, hp_reaction), hp_reaction] = player
                if connect(hp_board, opponent):
                    heuristic = -np.infty
                    return hp_reaction, heuristic
                else:
                    level_search(hp_board, opponent, (depth-1, heuristic))
    return


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
