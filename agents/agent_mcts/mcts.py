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
        self.parent = parent            # parent is a node or string
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

            # Checking if the child is an end-node, should be in the playout method.

            child = Node(child_board, opponent(node.player), parent=node)  # assign to the child, the parent
            node.children[action] = child                                  # assign to the parent, the child

            node.unexpanded.remove(action)

        return child

    @staticmethod
    def playout(node: Node) -> (Node, GameState, BoardPiece):
        """
        Simulation of the game is carried out till it's end.
        Use random agent.
        1. which moves are legal?
        2. choose random move.
        3. update game state (apply action, check for end, player switch etc.)
        :argument: node
        :return: Triple of a starting (last) node, a (win or draw) and the player, who moved last.
        """
        # maydo: use human_vs_agent function to playout random vs random.

        from agents.agent_random.random import generate_move_random
        player_new = node.player.copy()
        board_new = node.board.copy()
        node._count = 0

        while check_end_state(board_new, opponent(player_new)) == GameState.STILL_PLAYING:
            action_new = generate_move_random(board_new, player_new, None)

            # It at least one is unexpanded, we can start playout.
            # Handling the case with mixture of expanded and unexpanded moves:
            if action_new in node.children:
                node = node.children[str(action_new)]
                node._count = 0

            board_new = apply_player_action(board_new, action_new, player_new)
            player_new = opponent(player_new)
            node._count += 1

        result = node, check_end_state(board_new, opponent(player_new)), player_new

        return result

    def backprop(self, node: Node, game_state: GameState, last_player: BoardPiece):
        """
        Todo: Evaluate UCB for children, after updating the parent (or adding 1 in equation before the update).

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

    def _select_next_child(self, parent: Node, c=1.42) -> Node:
        """
        UCB - upper confidence bound - the bigger, the better.
        It's values will be used in the select method, but computed during backpropagation.
        When choosing from a parent node, it's child will have ucb stored.
        .
        At each node we need:
        w_i: wins at that node.
        s_i: (simulations) trials made from this node.
        s_p: parents trials to preserve power law across the tree, balancing exploration / exploitation.
        c : exploration parameter, np.sqrt(2) (or use rough approximation, for example 1.42)
        "argument_max(w_i/s_i + c*np.sqrt(np.ln(s_p)/s_i))"

        :argument: node
        :return: child

        Todo: Think of storing ucb in nodes (spreading the vector across children). Then updating only new ucb.
        Maydo: move the part below to tests.
        Note for the implementing me.
            Checking the boarder cases:
            - TODO: when the leaf has empty unexpanded set!!!!!
            - leaf with a full board or almost full - should be checked in the expand method,
            - when trials are zero,
            - when wins are zero,
            - when tree contains only root.
        maydo: Calculate statistics of each column-action (or each pair (action, nr count of move),
         (Confidence Interval?).
        """
        # prepare wins, trials data (parent trials are prepared):
        wins = np.zeros(7)
        trials = np.full(7, np.infty)  # if no child, we divide by infinity and get zero ucb value.
        for action_str, node in parent.children.items():
            wins[int(action_str)] = node.wins
            trials[int(action_str)] = node.trials
        trial_inverse = np.divide(1, trials)
        if parent.trials != 0:
            ucb_vector = wins*trial_inverse + np.sqrt(np.log(parent.trials)*trial_inverse)*c
        else:
            return parent.children[np.random.choice(list(map(int, parent.children.keys())))]
        if len(set(ucb_vector)) == 1:
            ucb = np.random.randint(3, 6)
        else:
            ucb = np.argmax(ucb_vector)
        print("\n\n", ucb, parent.children)
        return parent.children[str(ucb)]

    def select(self):
        """
        Policy to move from the root (current state of the game),
        to the optimal leaf of the tree.
        After using this function, simulation (playout) should be called from the child,
        which is the leaf.

        :return:
        """
        the_child = self.root
        while the_child.unexpanded == set():
            #available_moves(the_child.board):
            #{} and the_child.trials , the_child.wins ??:
            # versus available moves?
            # versus the child's children?

            the_child = self._select_next_child(the_child)
        return the_child


def generate_move_mcts(
        board_8: np.ndarray, player: BoardPiece, state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    After playout if node._count is still 0, then there was a win or draw in the expand call.
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
