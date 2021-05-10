from .minimax import generate_move_minimax as generate_move
from ..common import initialize_game_state, BoardPiece

board = initialize_game_state()
board[0, 0] = BoardPiece(0)