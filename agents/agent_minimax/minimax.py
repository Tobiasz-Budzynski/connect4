import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, PlayerAction, SavedState, GameState, lowest_free, apply_player_action, check_end_state


def generate_move_minimax(
        board_0: np.ndarray, player: BoardPiece, state: Optional[SavedState]
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
    # TODO: fix the bug of applaying every possible action at once.
    # TODO: check the loops and logic gates.
    # TODO: implement a proper alpha-beta pruning structure.
    # maydo: checking for immediate win could be as a first iteration.

    if state is None:
        state = SavedState(2,               # max depth
                           0,               # count depth
                           [0, None],       # current heuristic, action
                           [0, None],       # best untill now heuristic, action
                           board_0
                           )

    heuristic = state.heuristic_for_action[0]
    opponent = BoardPiece((player % 2) + 1)

    # ugly dealing with a bug, that spams the whole row with board pieces in the original board
    if state.count_depth == 0:
        state.board_5 = board_0

    # action (hp stands for hypothetical)
    for hp_action in range(7):
        hp_action = PlayerAction(hp_action)
        if lowest_free(board_0, hp_action) < 6:
            hp_board = apply_player_action(board_0, hp_action, player, copy=True)

            hp_game_state = check_end_state(hp_board, player)
            if hp_game_state == GameState.IS_DRAW:
                state.heuristic_for_action = (0, hp_action)

            # Opponents reaction
            if hp_game_state == GameState.STILL_PLAYING:
                for hp_reaction in range(7):
                    hp_reaction = PlayerAction(hp_reaction)
                    if lowest_free(hp_board, hp_reaction) < 6:
                        hp_re_board = apply_player_action(hp_board, hp_reaction, opponent, copy=True)

                        hp_re_game_state = check_end_state(hp_re_board, opponent)

                        if hp_re_game_state == GameState.IS_WIN:
                            break
                            # opponent would win, abort this action

                        # recursion
                        if (hp_re_game_state == GameState.STILL_PLAYING) and (state.count_depth != state.depth):
                            state.count_depth += 1
                            generate_move_minimax(hp_re_board, player, state)

                        # end of searched depth tree
                        else:
                            heuristic_norm = 7 ** (state.depth+1)
                            new_heuristic = heuristic / heuristic_norm

                            # good value
                            if new_heuristic > state.heuristic_for_action[0]:
                                # saving heuristic we normalize it
                                state.max_heuristic_for_action = [state.heuristic_for_action[0]/heuristic_norm,
                                                                  state.heuristic_for_action[1]]
                                # prepare for checking other options
                                state.heuristic_for_action = [0, None]
    # end of checking
    if state.max_heuristic_for_action[1] is None:
        # if You got here, it should be a lost game
        from agents.agent_random.random import generate_move_random
        return generate_move_random(state.board_5, player, None)
    return state.max_heuristic_for_action[1], state
    # TODO: check if the cases, when hypothetically we get a draw, don't lead here

