import numpy as np
from agents.common import BoardPiece, PlayerAction, GameState, SavedState, \
    apply_player_action, available_moves, opponent, check_end_state

from typing import Optional, Tuple


# Monte Carlo Tree Search
# The most basic way to use playouts
# is to apply the same number of playouts after each legal move
# of the current player, then choose the move
# which led to the most victories


class Node:
    """
    Possibly a class for node info:
        - Wins vs number of simulations.
        - player to move
        - state of the board
        - parent, children
    """

    def __init__(self, board: np.ndarray, player: BoardPiece, parent="root"):
        self.board = board
        self.player = player
        self.parent = parent            # parent is a node
        self.unexpanded = set(available_moves(board))
        self.children = {}              # {"action": new_node}
        self.wins = 0
        self.trials = 0


class MCTS:
    # class for MCTS, storing root etc.
    """
    Selection, expansion, playout, backpropagation.
    We define methods for Monte Carlo Tree Search.
    They should be called in this order in a loop,
    giving a best evaluated move - ratio wins/trials starting from the root.
    """

    def __init__(self, root):
        if isinstance(root, Node):
            self.root = root
        else:
            raise TypeError

    def select(self, child):
        """
        Use a policy to move from the root (current state of the game),
        to the optimal leaf of the tree.
        Afterwards, simulation (rollout) is called.
        wᵢ : this node’s number of simulations that resulted in a win
        sᵢ : this node’s total number of simulations
        sₚ : parent node’s total number of simulations
        c : exploration parameter
        :argument: child, (parent, if not accesiible through child) player to move,
        :return: a node, player to move, board representation.
        c = np.sqrt(2) (or use 1.42)
        argument_max(w/s + c*np.sqrt(np.ln(s_parent)/s))
        note for Implementing_me, maydo: Check the boarder cases (leaf is has a full board or almost full). Good luck!
        """
        raise NotImplementedError

    @staticmethod
    def expand(node: Node):
        """
        Make a child node from the node, randomly selecting from available actions.
        In child node store the new player to move, the state of the board and the parent.
        In parent node store reference to child, and delete the action taken from unexpanded.
        :param node: a leaf of the tree and future parent
        :return: nothing?  the child node?
        """
        if node.unexpanded == set():
            raise Exception('Can not choose an action from an empty set in node.unexpanded. You got to an end node.')

        else:
            action = np.random.choice(list(node.unexpanded))
            child_board = apply_player_action(node.board, action, node.player, copy=True)

            child = Node(child_board, opponent(node.player), parent=node)  # assign to the child, the parent
            node.children[action] = child                                  # assign to the parent, the child

            node.unexpanded.remove(action)
        return child

    @staticmethod
    def playout(node: Node) -> (GameState, BoardPiece):
        """
        Simulation of the game is carried out till it's end.
        Use random agent.
        1. which moves are legal?
        2. choose random move.
        3. update game state (apply action, check for end, player switch etc.)
        :argument: node
        :return: Tuple of a (win or draw) and the player, who moved last.
        """
        # maydo: use human_vs_agent function to playout random vs random.

        # Checking the node given.
        state = check_end_state(node.board, node.player)
        if state is not GameState.STILL_PLAYING:
            node.unexpanded = set()
            result = state, opponent(node.player)

        else:
            from agents.agent_random.random import generate_move_random
            player_new = node.player.copy()
            board_new = node.board.copy()

            from agents.common import pretty_print_board
            node._count = 0

            while check_end_state(board_new, opponent(player_new)) == GameState.STILL_PLAYING:
                action_new = generate_move_random(board_new, player_new, None)
                board_new = apply_player_action(board_new, action_new, player_new)
                player_new = opponent(player_new)
                node._count += 1

            result = check_end_state(board_new, opponent(player_new)), player_new

        return result

    def backprop(self, node: Node, game_state: GameState, last_player: BoardPiece):
        """
        Backprop increases the trials count along the path to the tree's root
        and accordingly to the end state - the wins of every second node along the path,
        (without changing wins, when it is a draw).
        If the winner is the node's player, then the opponent nodes along the path
         get wins increased (because the statistics is used, by the parent).
        If the winner is the node's opponent, then the node's wins is increased.
        :return:
        """
        if not node == "root":
            node.trials += 1
            if last_player == opponent(node.player):
                node.wins += 1
            self.backprop(node.parent, game_state, last_player)

        elif node == "root":
            pass


def generate_move_mcts(
        board_8: np.ndarray, player: BoardPiece, state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

    """
    # for
    #    for trial in range(SavedStateMCTS.trials_number.value):


class SavedStateMCTS(SavedState):
    def __init__(self, tree):
        """

        :rtype: object
        """
        self.trials_number = 10  # per possible move
        self.tree = tree
