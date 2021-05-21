import numpy as np
from typing import Optional, Tuple
from agents.common import lowest_free, BoardPiece, PlayerAction, SavedState

import timeit  # time checking
#import cProfile  # for profiling executing di
#from numba import njit  # compiler optimizing loops, decorater.

#class SavedState:
#    def __init__(self, computational_result):
#        self.computational_result = computational_result


def generate_move_random(
     board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
     ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose a valid, non-full column randomly and return it as `action`.
    """
    low_frees = np.zeros(board.shape[1])
    for j in range(board.shape[1]):
        low_frees[j] = lowest_free(board, j)
    columns_free = np.argwhere(low_frees < board.shape[0]).reshape(-1)
    action = np.random.choice(columns_free).astype(PlayerAction)
    return action, saved_state