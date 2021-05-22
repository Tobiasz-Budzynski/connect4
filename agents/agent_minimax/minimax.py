import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, PlayerAction, SavedState, lowest_free, connect


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, state: Optional[SavedState]
        ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    The function checks if action leads to win, if Yes it returns it,
    if No it avoids opponents wins. For each opponent response a heuristic[0] is calculated.
    It increments, when there is a win opportunity for the player.
    At the end of the function heuristic[0] is normalized according to depth,
    to have values from 0 to 1 (sth like a likely-hood of winning),
    unless heuristic[0] is -infinity, which means that opponent can force a win.
    Heuristic[1] is just an action considered by the player.
    """
    if state is None:
        state = SavedState(2, 0, [0, None], [0, None])  # depth, count_depth,
    depth = state.depth - state.count_depth
    heuristic = state.heuristic_for_action[0]
    opponent = (player % 2) + 1
    hp_board = np.copy(board)  # abbreviation of hypothetical
    for hp_action in np.ndarray([3, 4, 2, 5, 1, 6, 0]).astype(np.int8):
        # maydo: chose the column with an order deterministic pseudo-gaussian-like centered at three.
        # ...and change the order after each game.
        hp_board[lowest_free(hp_board, hp_action), hp_action] = player
        if state.count_depth == 0:
            state.heuristic_for_action[1] = hp_action
            # save a current hp_action considered and pair it with heuristic
        if connect(hp_board, player, n=4):
            if depth == state.depth:
                state.heuristic_for_action[0] = 1  # here it is already normalized
                state.count = state.depth
                return state.heuristic_for_action[1], state
            else:
                heuristic += 7**depth
        else:
            hp_reaction = None
            for hp_reaction in np.ndarray([3, 4, 2, 5, 1, 6, 0]).astype(np.int8):
                hp_board[lowest_free(hp_board, hp_reaction), hp_reaction] = player
                if connect(hp_board, opponent):
                    break
                    # when opponent wins in this move, abort the action You considered
                    # implement alpha beta pruning

            if state.count_depth != state.depth:
                state.count_depth += 1
                generate_move_minimax(hp_board, opponent, state)
            else:
                heuristic_norm = 7 ** (state.depth+1)
                new_heuristic = heuristic / heuristic_norm
                if new_heuristic > state.heuristic_for_action[0]:
                    # saving heuristic we normalize it
                    state.max_heuristic_for_action = [state.heuristic_for_action[0]/heuristic_norm,
                                                      state.heuristic_for_action[1]]
                    # prepare for checking other options
                    state.heuristic_for_action[0] = [0, None]
                    # end of checking
                if hp_reaction == 0:
                    return state.max_heuristic_for_action[1], state
    # if You got here, it's a lost game
    from agents.agent_random import generate_move
    return generate_move(board, player, None)
