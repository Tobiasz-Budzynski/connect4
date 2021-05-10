import numpy as np
import timeit  # time checking
#import cProfile  # for profiling executing di
#from numba import njit  # compiler optimizing loops, decorater.

from typing import Optional, Tuple
from agents.common import lowest_free, BoardPiece, PlayerAction, SavedState


class SavedState:
    def __init__(self, computational_result):
        self.computational_result = computational_result


def generate_move_random(
     board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
     ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose a valid, non-full column randomly and return it as `action`.
    """
    low_frees = np.zeros(board.shape[1])
    for i in range(board.shape[1]):
        low_frees[i] = lowest_free(board, i)
    columns_free = np.argwhere(low_frees < board.shape[0] + 1).reshape(-1)
    action = np.random.choice(columns_free).astype(PlayerAction)
    print(columns_free)
    saved_state = None

    return action, saved_state