import numpy as np

from agents.common import GenMove, BoardPiece
from agents.agent_minimax.minimax import level_search

def test_level_search():
    from ..test_common import prepare_board_for_testing
    board = prepare_board_for_testing()
    player = np.int8(np.random.randint(2)).astype(BoardPiece)
    assert level_search(board, player, None) in [GameState.IS_WIN or None]
    #assert type(level_search) == GenMove