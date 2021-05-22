import numpy as np

from agents.common import BoardPiece, SavedState
from agents.agent_minimax.minimax import generate_move_minimax


def test_generate_move_minimax():
    from ..test_common import prepare_board_for_testing
    board, low_frees = prepare_board_for_testing()
    player = 1 + np.int8(np.random.randint(2)).astype(BoardPiece)
   # state = SavedState(2, 0, [0, None], [0, None])  # depth, count_depth,
                                        # heuristic for action, max heuristic for action
    a = generate_move_minimax(board, player, None)