import numpy as np

from agents.common import GenMove, BoardPiece
from agents.agent_minimax.minimax import level_search, SavedMM


def test_level_search():
    from ..test_common import prepare_board_for_testing
    board, low_frees = prepare_board_for_testing()
    player = 1 + np.int8(np.random.randint(2)).astype(BoardPiece)
    state = SavedMM(2, 0, [])
    a = level_search(board, player, state)
    #assert type(level_search) == GenMove