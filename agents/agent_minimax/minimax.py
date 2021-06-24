import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, PlayerAction, SavedState, GameState, lowest_free, apply_player_action, check_end_state, GameDimensions

# MOnte CArlo Tree Search
# Few classes to structure the search

def heuristic_func(depth_count: int, game_state: GameState) -> float:
    """"
    Knowing which depth we are considering we can deduce, if it is an opponent or a player.
    Accordingly, we give heuristics positive or negative.
    1st version is with just wins and loses normalized according to depth.
    """



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
    # TODO: implement a proper alpha-beta pruning structure.
    # maydo: checking for immediate win could be as a first iteration.

    # Remark: this function is way too long. Generally, functions should only do one specific thing, which makes them
    #         easier to understand, maintain, and test. So you should heavily refactor this function and split it up.

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

    possible_moves = []
    for hp_action in range(GameDimensions.LENGTH.value):
        if lowest_free(board_0, hp_action) < 6:
            PlayerAction(possible_moves.append(hp_action))


    # Remark: In general, you should refactor this part. You have a lot of repeated stuff below, which you can
    #         bring into a more concise form
    # Remark: So far, you're only checking for wins/losses/draws. You need to make the transition to true minimax!
    for hp_action in possible_moves:

        hp_board = apply_player_action(board_0, hp_action, player, copy=True)
        hp_game_state = check_end_state(hp_board, player)
        if hp_game_state == GameState.IS_DRAW:
            state.heuristic_for_action = (0, hp_action)
        if hp_game_state == GameState.IS_WIN:
            # TODO: use heuristic func
            state.heuristic_for_action = (heuristic_func(), hp_action)
        if (hp_re_game_state == GameState.STILL_PLAYING) and (state.count_depth != state.depth):
                            state.count_depth += 1
                            generate_move_minimax(hp_re_board, player, state)
                            # Remark: it's a bit weird that you start recursion only here, and not in the first loop.

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

