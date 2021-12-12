import random
import time

import simulate_game
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from math import isqrt
from operator import itemgetter
from copy import deepcopy
from collections import Counter
from itertools import groupby


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        self.scoremap = {0: 0,
                         1: 1,
                         2: 3,
                         3: 7}
        super().__init__()

    def countfilled(self, squares):
        """

        @param squares: the squares parameter from the game state
        @return: the total amount of filled columns, rows and blocks

        The function takes an array of length N*N and computes, when converted to a sudoku board, how many rows, columns
        and blocks don't contain any filled-in values. it returns the sum of the amount of rows, columns and blocks.
        """
        # split the data into groups representing rows and columns
        row_split = [squares[x:x + self.N] for x in range(0, len(squares), self.N)]
        col_split = [itemgetter(*[(self.N * i) + j for i in range(self.N)])(squares) for j in range(self.N)]

        # split the data into groups representing blocks
        indices_block = [self.N * (j // self.n) + j % self.n + (i * self.n) % self.N + (i // self.m) * self.N * self.m
                         for i in range(self.N) for j in range(self.N)]

        split_indices_block = [indices_block[x:x + self.N] for x in range(0, len(indices_block), self.N)]
        block_split = [itemgetter(*index)(squares) for index in split_indices_block]
        # count all the empty values for each group in the rows, columns and blocks, then take the total of the three.
        return sum([[sublist.count(0) for sublist in zone].count(0) for zone in [row_split, col_split, block_split]])

    def find_legal_moves(self, board):
        """"
        finds all legal moves by going through every value and checking if there is a value.
        If this is the case it looks at the row, column and block where the value is at and removes the value from the
        possible moves stored in the row, column and block.
        """

        # list of possible moves
        rows = {i: list(range(1, (self.N + 1))) for i in range(self.N)}
        cols = {i: list(range(1, (self.N + 1))) for i in range(self.N)}
        blocks = {i: list(range(1, (self.N + 1))) for i in range(self.N)}

        def remove_illegal(i, j):
            """
            for a value located at (i,j). If the value is filled in (>0) remove it from the lists of posible moves
            corresponding to that row, column and block.
            """
            value = board.get(i, j)
            if value != 0:
                rows[j].remove(value)
                cols[i].remove(value)
                blocks[self.m * (i // self.m) + j // self.n].remove(value)
            return

        # remove all illegal moves
        [remove_illegal(i, j) for i in range(self.N) for j in range(self.N)]

        # a move (i,j) is only legal if it is legal in the column row and block, so compute the intersect of them.
        legal_moves = {(i, j): set(rows[j]).intersection(cols[i], blocks[self.m * (i // self.m) + j // self.n])
                       for i in range(self.N)
                       for j in range(self.N)}

        return legal_moves

    def compute_best_move(self, game_state: GameState) -> None:
        # parameters
        self.N = game_state.board.N
        self.rootn = isqrt(self.N)
        self.N2 = self.N*self.N
        self.n = game_state.board.n
        self.m = game_state.board.m
        # find all legal moves (according to sudoku rules)
        legal_moves = self.find_legal_moves(board=game_state.board)

        # find all hidden singles
        """
        block_lookup_table = {i: [] for i in range(self.N)}
        [block_lookup_table[self.m * (i // self.m) + j // self.n].append((i, j))
         for i in range(self.N) for j in range(self.N)]

        for i in range(self.N):
            for j in range(self.N):
                current_move = legal_moves[(i, j)]
                if len(current_move) > 1:
                    row_moves = [legal_moves[(i, row)] for row in range(self.N) if row != j]
                    row_hsingle = set(current_move) - set.union(*row_moves)
                    if len(row_hsingle) == 1:
                        legal_moves[(i, j)] = row_hsingle
                        continue

                    col_moves = [legal_moves[(col, j)] for col in range(self.N) if col != i]
                    col_hsingle = set(current_move) - set.union(*col_moves)
                    if len(col_hsingle) == 1:
                        legal_moves[(i, j)] = col_hsingle
                        continue

                    block_moves = [legal_moves[pos] for pos in
                                   block_lookup_table[self.m * (i // self.m) + j // self.n] if pos != (i, j)]
                    block_hsingle = set(current_move) - set.union(*block_moves)
                    if len(block_hsingle) == 1:
                        legal_moves[(i, j)] = block_hsingle
                        continue
        """

        def same_block(bin):
            blocks = [bin_i/isqrt(self.N) + bin_i % isqrt(self.M) for bin_i in bin]
            if(len(set(blocks)) == 1):
                return True
            else:
                return False

        def get_block_moves(bin_i):
            block = bin_i/isqrt(self.N) + bin_i % isqrt(self.M)
            startIndex = block * self.rootn
            moves = {}
            for i in range(self.M):
                for k in range(self.N):
                    moves[(i, k)] = (legal_moves[(i, k)])
            return moves

        for i in range(self.N):
                row_moves = [legal_moves[(i, row)] for row in range(self.N)]
                if len(row_moves) > 1:
                    v_bins = []
                    for v in range(self.N):
                        v_bin = [(i * self.N + key) for key, row in row_moves if v in row]
                        v_bins.append(v_bin)
                    for bin_val, bin in enumerate(v_bins):
                        if(same_block(bin)):
                            """ Cross out posibilities """
                            cross_moves = get_block_moves(bin[0])
                            actual_to_cross = [val for ind, val in cross_moves if (ind[0] * self.N + ind[1]) not in bin]
                            for index, moves in actual_to_cross:
                                legal_moves[index] = set(moves) - set(Move(index[0], index[1], bin_val))

        def possible(i, j, value, game_state):
            # find only moves for empty squares and non-taboo, legal moves.
            return game_state.board.get(i, j) == SudokuBoard.empty and not\
                TabooMove(i, j, value) in game_state.taboo_moves and \
                value in legal_moves[(i, j)]

        # find all non forfeiting moves.
        all_moves = [(Move(i, j, value), (i * self.N) + j)
                     for i in range(self.N)
                     for j in range(self.N)
                     for value in range(1, self.N + 1)
                     if possible(i, j, value, game_state)]


        # assuming that filling in a value which is most common on the board is more likely to keep the sudoku solveable
        # we only keep, for each indice where we might play a move, the move corresponding to the most common number
        # on the board that is playable.
        most_common_numbers = Counter(game_state.board.squares)
        new_moveset = []
        for group in groupby(all_moves, lambda x: x[1]):
            moveset = list(group[1])
            candidate = moveset[0]
            for move in moveset:
                if most_common_numbers[move[0].value] > most_common_numbers[candidate[0].value]:
                    candidate = move
            new_moveset.append(candidate)
        all_moves = new_moveset

        # calculate various additional parameters for each move
        def calcmove(indice, prev_score, calcsquares, calcname):
            """

            @param indice:  where on the sudoku board the move will take place
            @param prev_score:  the score for the board prior to the move being played
            @param calcsquares: a 1D array of the board before prior to the move being played
            @param calcname: the name of the parent of this node
            @return:
                nsquares: a 1D array of the board after the move has been played
                scorediff: the amount of points scored by playing the move
                score: the amount of rows,columns and blocks that are filled after the move has been played
                calc_children: a list of the moves that can be played after playing this move

            Calculates the state of the board and score after playing a hypothethical move.
            """
            nsquares = calcsquares.copy()
            # -1 is used to denote a move has been played on a square without needing to specify what the number is.
            nsquares[indice] = -1
            score = self.countfilled(nsquares)
            scorediff = score - prev_score

            # find children
            indices = [i for i, j in enumerate(nsquares) if j == 0]
            ccountr = 0
            calc_children = []
            for indice in indices:
                # the name for each possible move is a list starting with the name of the parent node (node directly
                # under the root) and from the positions under each of the parents node children, grandchildren etc.
                cname = calcname + [ccountr]
                ccountr += 1
                calc_children.append({'move': indice, 'name': cname})
            return nsquares, self.scoremap[scorediff], score, calc_children

        # calculate the score for the root position.
        initial_score = self.countfilled(game_state.board.squares)


        def get_candidate_node(tag, move):
            """

            @param tag: a unique identifier
            @param move: a move object
            @return: a dictionary representing a move which contains all important parameters for the move

            Calculates all important parameters for the direct children of the root nodes and returns them as a dict.
            Important is that 'eval' and 'tally' are the same but during the minimax eval will house the children of the
            node whilst tally stays unchanged.
            """
            name = [f'p{tag}']
            squares, points, score, can_children = calcmove(move[1], initial_score, game_state.board.squares, name)
            return {'name': name, 'move': move[0], 'squares': squares, 'eval': points, 'tally': points, 'score': score,
                    'children': can_children}

        # construct a tree as a collection of the direct children of the root node
        tree = [get_candidate_node(nr, move) for nr, move in enumerate(all_moves)]

        # get best candidate for depth=1:
        proposed_move = max(tree, key=lambda x: x['eval']).copy()
        self.propose_move(proposed_move['move'])



        def get_general_node(child,parent):
            """

            @param child: a child of some node
            @return:
                as_node: returns a dict containing only the name (which is equal to its position within the tree) and
                the evaluation at that position
                as_move: returns a dict containing all information which is needed to compute the as_node parameters for
                the children of this node.

            This function computes the evaluation on a node given its parent and returns the evaluation along with
            the necessary parameters to compute the eval for its children.
            """

            child_squares, child_points, child_score, child_children = \
                calcmove(child['move'], parent['score'], parent['squares'], child['name'])

            # the eval represents the points lead gained by the player. since the name represents the position of the
            # node in the tree, the lenght of the name represents the depth and thus if it is the opponents turn or
            # players turn. For even lenghts the opponent moves and the evaluation therefore goes down by the points
            # the opponent scores.
            if len(child['name']) % 2 == 0:
                child_eval = parent['tally'] - child_points
            else:
                child_eval = parent['tally'] + child_points

            as_node = {'name': child['name'], 'eval': child_eval}
            as_move = {'name': child['name'],
                       'squares': child_squares,
                       'score': child_score,
                       'children': child_children,
                       'tally': child_eval,
                       'eval': child_eval}
            return as_node, as_move

        # initialize the parameter moves as the set of children of the root node.
        moves = tree
        # depth is a cosmetic parameter
        depth = 0

        def place_children(children, parent):
            child_nodes = list(zip(*children))[0]

            insertion_pos = next(x for x in tree if x['name'][0] == parent['name'][0])
            for nesting in parent['name'][1:]:
                insertion_pos = insertion_pos['eval'][nesting]
            insertion_pos['eval'] = child_nodes

            return list(zip(*children))[1]

        while True:
            ########################################
            # populating the tree with a new layer #
            ########################################
            # initializing parameters
            depth += 1
            proposed_move['eval'] = float('-inf')
            #allchildren = []

            # at most N*N layers can be computed so after this it does not make sense to keep the loop running.
            if depth > self.N2:
                break
            print(f'search depth:{depth}')
            # compute the evaluation function for the children of each leaf in the current tree and add them as leaves.
            moves_nested = [place_children([get_general_node(child, parent) for child in parent['children']],
                            parent) for parent in moves if len(parent['children']) > 0]
            moves = [package for move_package in moves_nested for package in move_package]

            """
            for parent in moves:
                if len(parent['children']) > 0:

                    children = [get_general_node(child) for child in parent['children']]
                    child_nodes = list(zip(*children))[0]
                    child_moves = list(zip(*children))[1]

                    allchildren = allchildren+list(child_moves)

                    # the new leaves are stored in the eval of the previous leaves, the name gives the position where
                    # the new leaf should be stored, this code looks through the name and then stores the new leaf
                    # in the right position of the tree
                    insertion_pos = next(x for x in tree if x['name'][0] == parent['name'][0])
                    for nesting in parent['name'][1:]:
                        insertion_pos = insertion_pos['eval'][nesting]
                    insertion_pos['eval'] = child_nodes
            """
            ##################################
            # back propagation using minimax #
            ##################################
            backcopy = deepcopy(tree)

            # perform minimax on the tree where if the node is a leaf the 'eval' is an integer representing the point
            # gain for the player since the root node, and otherwise the 'eval' is a list of the children of this node.
            def minimax(movetree, alpha, beta):
                if type(movetree['eval']) == int:
                    return movetree['eval']

                # same trick as before, since the name represents the position the length represents the depth, the
                # polarity therefore represents whether it is a minimizing or maximizing layer.
                if len(movetree['name']) % 2 == 0:
                    max_eval = float('-inf')
                    for child in movetree['eval']:
                        eval = minimax(child, alpha, beta)
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
                    return max_eval
                else:
                    min_eval = float('inf')
                    for child in movetree['eval']:
                        eval = minimax(child, alpha, beta)
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
                    return min_eval

            for move in backcopy:
                # evaluate all possible moves and find the best one.
                move['eval'] = minimax(move, float('-inf'), float('inf'))
                if move['eval'] > proposed_move['eval']:
                    proposed_move = move
                    print(f'move:{proposed_move["move"]} and evaluation: {proposed_move["eval"]}')
                    self.propose_move(move['move'])

            # assign all of the new leafs to be analyzed for the next population step.
            #moves = allchildren
            # if there are not more children then stop running the loop.
            if len(moves) == 0:
                break


