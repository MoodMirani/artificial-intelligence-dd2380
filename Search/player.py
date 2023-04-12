import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives anfrom play
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
        Main loop for the ab_minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:

            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node, player=0)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node, player):
        """
        Use ab_minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING ab_MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method! 
        # compute_and_get_children() returns a list of the child Nodes, found one level below in the game tree.

        startTime = time.time()
        timeLimit = 0.055
        timeout = False
        alpha = float('-inf')
        beta = float('inf')
        currentValue = float('-inf')                              
        depth = 0
        nextMove = 0
        children = initial_tree_node.compute_and_get_children()
        children.sort(key=self.heuristic_eval, reverse=True)

        # IDS
        while not timeout:
            try:
                for child in children:
                    value = self.ab_minimax(child, player, depth, alpha, beta, startTime, timeLimit)
                    if value > currentValue:
                        currentValue = value
                        nextMove = child.move
                depth += 1
                if (time.time() - startTime) > timeLimit:
                    raise TimeoutError
            except:
                timeout = True
        return ACTION_TO_STR[nextMove]

    def ab_minimax(self, node, player, depth, alpha, beta, startTime, timeLimit):
        # move ordering
        children = node.compute_and_get_children()
        children.sort(key=self.heuristic_eval, reverse=True)

        if (time.time() - startTime) > timeLimit or len(children) == 0 or depth == 0:
            return self.heuristic_eval(node)
        
        elif player == 0:
            value = float('-inf')
            for child in children: # children are positions that can be reached
                value = max(value, self.ab_minimax(child, 1, depth - 1, alpha, beta, startTime, timeLimit))
                alpha = max(alpha, value)
                if beta <= alpha: # β prune
                    return value

        elif player == 1:
            value = float('inf')
            for child in reversed(children):
                value = min(value, self.ab_minimax(child, 0, depth - 1, alpha, beta, startTime, timeLimit))
                beta = min(beta, value)
                if beta <= alpha: # α prune
                    return value
        return value

    def heuristic_eval(self, node):
        player_score, opponent_score = node.state.get_player_scores() 
        score = player_score - opponent_score

        for fish_index, fish_pos in node.state.get_fish_positions().items():
            distance = abs((node.state.get_hook_positions()[node.state.player][0] - fish_pos[0]) + (node.state.get_hook_positions()[node.state.player][1] - fish_pos[1]))
            if distance == 0:                                                                                           
                score += (1 - node.state.player) * node.state.get_fish_scores()[fish_index]                 
            else:
                score += ((1 - node.state.player) * node.state.get_fish_scores()[fish_index]) / distance    
        return score



