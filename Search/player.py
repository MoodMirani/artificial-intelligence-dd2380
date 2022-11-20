import random
import math
import time

# from operator import itemgetter

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return

class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        self.repeated_states_dict = {}
        super(PlayerControllerMinimax, self).__init__()


    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:

            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            bestMove = self.search_best_next_move(initial_tree_node=node, player=0)

            # Execute next action
            self.sender({"action": bestMove, "search_time": None})

    def search_best_next_move(self, initial_tree_node, player):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method! 
        # compute_and_get_children() returns a list of the child Nodes, found one level below in the game tree.

        startTime = time.time()
        timeLimit = 0.07
        timeout = False
        alpha = -math.inf
        beta = math.inf
        startValue = -math.inf                               
        depth = 0
        bestMove = 0
        children = initial_tree_node.compute_and_get_children()

    
        # children.sort(key=self.heuristic_eval, reverse=True)

        # IDS
        while not timeout:
            try:
                for child in children:
                    value = self.minimax(child, player, depth, alpha, beta, startTime, timeLimit)
                    if value > startValue:
                        startValue = value
                        bestMove = child.move
                depth += 1
                if (time.time() - startTime) > timeLimit:
                    raise TimeoutError
            except:
                timeout = True
        return ACTION_TO_STR[bestMove]


    def hash(self, state):
        '''
        Computes the string hash of a given state, by describing state in full
        :param state:
        :return: hashed string of state
        '''
        string = str(state.get_player_scores()) + str(state.get_fish_positions()) + str(state.get_hook_positions())
        hashed_string = hash(string) # 32 bit istället 64 bit
        return hashed_string

    def minimax(self, node, player, depth, alpha, beta, startTime, timeLimit):
        '''
        MINIMAX FUNCTION with alpha-beta pruning
        :param node: node that we are analysing
        :param player: player that is playing and decides if we want to find min or max
        :param depth: depth of node
        :param alpha:
        :param beta:
        :param startTime:
        :param timeLimit:
        :return: a heuristic value that approximates a utility function of the state
        '''

        if (time.time() - startTime) > timeLimit:
            raise TimeoutError

        # Repeated state check
        key = self.hash(node.state)
        if key in self.repeated_states_dict and self.repeated_states_dict[key][0] >= depth:
            return self.repeated_states_dict[key][1]

        # move ordering
        children = node.compute_and_get_children()
        children.sort(key=self.heuristic_eval, reverse=True)


        if (time.time() - startTime) > timeLimit or len(children) == 0 or depth == 0:
            return self.heuristic_eval(node)

        elif player == 0:
            v = float('-inf')
            for child in children:
                v = max(v, self.minimax(child, 1, depth - 1, alpha, beta, startTime, timeLimit))
                alpha = max(alpha, v)
                if (time.time() - startTime) > timeLimit or beta <= alpha:
                    return v

        elif player == 1:
            v = float('inf')
            for child in reversed(children):
                v = min(v, self.minimax(child, 0, depth - 1, alpha, beta, startTime, timeLimit))
                beta = min(beta, v)
                if (time.time() - startTime) > timeLimit or beta <= alpha:
                    return v


        self.repeated_states_dict.update({key: [depth, v]})
        return v

    def heuristic_eval(self, node):
        '''
        :param node: node
        :return: returns score
        '''
        player = node.state.player
        player_score, opponent_score = node.state.get_player_scores() # Spelarnas poäng

        score = player_score - opponent_score
        fish_scores = node.state.get_fish_scores()                  # Fiskarnas score
        fish_positions = node.state.get_fish_positions()            # Fiskarnas position
        hook_position = node.state.get_hook_positions()[player]     # krokposition för player

        for fish_index, fish_pos in fish_positions.items():
            distance = abs(fish_pos[0] - hook_position[0]) + abs(fish_pos[1] - hook_position[1]) # manhattan
            if distance == 0:                                                                    # distance == 0, fångade fisk
                score += (1 - (player * 2))*(fish_scores[fish_index])*10                         # fångade fiskar väger mer
            else:
                score += (((1 - (player * 2))*(fish_scores[fish_index]) / (distance)))     # ej delat med 0, vid player 0 = +, player 1 = -

        return score



