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
    Selection, expansion, playout, backpr`opagation.
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
            if len(available_moves(node.board)) == 0:
                return node
            else:
                raise Exception("\n The unexpanded set is empty, while still there are available moves. \n The select method shouldn't have alowed it.")

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

            # If at least one is unexpanded, we can start playout.
            # Handling the case with mixture of expanded and unexpanded moves:
            if action_new in node.children:
                node = node.children[action_new]
                node._count = 0

            board_new = apply_player_action(board_new, action_new, player_new)
            player_new = opponent(player_new)
            node._count += 1

        result = node, check_end_state(board_new, opponent(player_new)), player_new

        return result

    def backprop(self, node: Node, game_state: GameState, last_player: BoardPiece):
        """
        Backprop increases the trials count along the path to the tree's root
        and accordingly to the end state - the wins of every second node along the path,
        (without changing wins, when it is a draw).
        If the winner is the node's player, then the opponent nodes along the path
         get wins increased (because the statistics is used, by the parent).
        If the winner is the node's opponent, then the node's wins is increased.
        node: is the leaf from which playout was made. Beware - nodes player is the one to make a move
        so it's the opponent of node's player that had the last move on node's board.
        game_state: the state after playout.
        last_player: is the player that made the last move in the playout.
        """
        if not node == "root":
            node.trials += 1

            if last_player == node.player:  # careful with who is the opponent, look commentary above
                node.wins += 1
            self.backprop(node.parent, game_state, last_player)

        elif node == "root":
            pass

    def _select_next_child(self, parent: Node, c=1.42) -> Node:
        """
        UCB - upper confidence bound - the bigger, the better for the node.
        It's values will be used in the select method.
        When choosing from a parent node, it's child will have ucb stored.

        At each node we need and compute:
        w_i: wins at that node.
        s_i: (simulations) trials made from this node.
        s_p: parents trials to preserve power law across the tree, balancing exploration / exploitation.
        c : exploration parameter, np.sqrt(2) (or use rough approximation, for example 1.42)
        "argument_max(w_i/s_i + c*np.sqrt(np.ln(s_p)/s_i) | i goes through siblings)"

        :argument: node
        :return: child with the highest heuristic value.

        maydo: Think of storing ucb in nodes (spreading the vector across children) and computing during backprop.
        Maydo: move the part below to tests.
        Note for the implementing me.
            Checking the boarder cases:
            - when the leaf has empty unexpanded set? A: when the node is fully expanded or when the node is at the Draw (and the tree was freshly started).
            - leaf with a full board or almost full - should be checked in the expand method,
            - when trials are zero,
            - when wins are zero,
            - when tree contains only root.
        maydo: Calculate statistics of each column-action (or each pair (action, nr count of move),
         (Confidence Interval?).

        Todo: Evaluate UCB for children, after updating the parent (or adding 1 in equation before the update).
        """
        # Prepare wins, trials data (parent trials are prepared).
        wins = np.zeros(7)
        trials = np.full(7, np.infty)  # if no child, we divide by infinity and get zero ucb value.
        for action_str, node in parent.children.items():
            wins[int(action_str)] = node.wins
            trials[int(action_str)] = node.trials
        trial_inverse = np.divide(1, trials)

        # Calculate the values.
        if parent.trials == 0:  # first trial
            return parent.children[np.random.choice(list(map(int, parent.children.keys())))]
        elif parent.trials != 0:
            ucb_vector = wins*trial_inverse + np.sqrt(np.log(parent.trials)*trial_inverse)*c
        ucb = np.argmax(ucb_vector)

        return parent.children[ucb]

    def select(self):
        """
        Policy to move from the root (current state of the game),
        to the optimal leaf of the tree.
        After using this function, simulation (playout) should be called from the child,
        which is the leaf.

        :return: the leaf with some unexpanded or an end game node
        """

        the_child = self.root
        while the_child.unexpanded == set() and not len(available_moves(the_child.board)) == 0:

            the_child = self._select_next_child(the_child)
        return the_child


def generate_move_mcts(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    The function unpack the tree from saved state.
    Run consecutively the methods of Monte Carlo Tree Search, to find the next action.
    After playout if node._count is still 0, then there was a win or draw in the expand call.
    :board: the games board at present.
    :player: the board piece to make a move.
    :saved_state: the instance of the saved state (might be None at first).

    :return: the action and the instance of the saved state, which stores the tree.

    maydo: Optimize nr of loops as a function of the current nr of pieces on board,
        so it passes a test of blocking the opponent in next move (avoiding opponent win in the next move).
    """

    # Unpack saved state.
    if saved_state is None:
        root = Node(board, player)
        t = MCTS(root)  # stands for a tree
        saved_state = SavedStateMCTS(t)
    else:
        t = saved_state.tree
        index_last_action = np.argwhere(board != t.root.board)
        t.root = t.root.children[int(index_last_action[:,1])]

    # With 3k loops it's really good, but slow.
    if np.count_nonzero(board) in range(6):
        nr_of_loops = 2000
    else:
        nr_of_loops = 1500

    # The MCTS usage.
    for loop in range(nr_of_loops):
        t.backprop(*t.playout(t.expand(t.select())))
    child = t._select_next_child(t.root)

    # From the node obtain the action.
    for key, value in t.root.children.items():
        if value == child:
            action = key
            break

    # Update situation saved.
    t.root = t.root.children[action]
    saved_state.tree = t

    return action, saved_state


class SavedStateMCTS(SavedState):
    def __init__(self, tree):
        """
        :rtype: object
        """
        self.tree = tree