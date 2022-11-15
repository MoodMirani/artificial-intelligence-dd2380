#!/usr/bin/env python3
import random
import math
import time

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
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def heuristic(self, initial_tree_node): #l1
        totalScore = initial_tree_node.state.player_scores[0] - initial_tree_node.state.player_scores[1]
        h = 0
        for i in initial_tree_node.fish_positions:
            distance = min(abs(initial_tree_node.fish_positions[0] - initial_tree_node.hook_position[0]), 20- abs(initial_tree_node.fish_positions[0] - initial_tree_node.hook_position[0])) + abs(initial_tree_node.fish_positions[1]- initial_tree_node.hook_position[1])
            if distance == 0 and initial_tree_node.fish_scores[i] > 0:
                return math.inf
            h = max(h, initial_tree_node.fish_scores[i]* math.exp(-distance))
        return 2 * totalScore + h

    def ab_minimax(self, initial_tree_node, state, depth, alpha, beta, player, startTime):
        if time.time() - startTime > 0.075:
            raise TimeoutError
        else:
            children = initial_tree_node.compute_and_get_children()
            if depth == 0 or len(children) == 0:
                return self.heuristic(initial_tree_node)

            elif player == 0:
                maxEval = -math.inf
                for child in state:
                    eval = self.ab_minimax(child, child.state, depth - 1, alpha, beta, player=1, startTime=startTime)
                    maxEval = max(maxEval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return maxEval
            else:

                minEval = +math.inf
                for child in state:
                    eval = self.ab_minimax(child, child.state, depth - 1, alpha, beta, player=0, startTime=startTime)
                    minEval = min(minEval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return minEval


    def search_best_next_move(self, initial_tree_node):
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
        startTime = time.time()
        timeout = False
        alpha = -math.inf
        beta = math.inf
        children = initial_tree_node.compute_and_get_children()
        scores = []
        depth = 0
        best_move = 0
        while not timeout:
            try:
                for child in children:
                    print('hej')
                    score = self.ab_minimax(child, child.state, depth, alpha, beta, 1, startTime)
                    scores.append(score)
                    depth += 1
                best_score = scores.index(max(scores))
                move = children[best_score].move

                best_move = move
            except:
                timeout = True

        random_move = random.randrange(5)
        return ACTION_TO_STR[best_move]
