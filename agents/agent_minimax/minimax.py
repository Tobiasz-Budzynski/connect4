import numpy as np
from typing import Optional
from ..common import BoardPiece, PlayerAction


# SavedState should be a class, dictionary, array or list?

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