import random
import time
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
        row_split = [squares[x:x + self.N] for x in range(0, len(squares), self.N)]
        col_split = [itemgetter(*[(self.N * i) + j for i in range(self.N)])(squares) for j in range(self.N)]

        multfac = int(self.N2 / self.rootn)
        indices_block = [self.N * (i // self.rootn % self.N) + i % self.rootn + i // multfac * self.rootn
                         for i in range(self.N2)]
        split_indices_block = [indices_block[x:x + self.N] for x in range(0, len(indices_block), self.N)]
        block_split = [itemgetter(*index)(squares) for index in split_indices_block]

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
                blocks[self.rootn * (i // self.rootn) + j // self.rootn].remove(value)
            return

        # remove all illegal moves
        [remove_illegal(i, j) for i in range(self.N) for j in range(self.N)]

        # a move (i,j) is only legal if it is legal in the column row and block, so compute the intersect of them.
        legal_moves = {(i, j): set(rows[j]).intersection(cols[i], blocks[self.rootn * (i // self.rootn) + j // self.rootn])
                       for i in range(self.N)
                       for j in range(self.N)}
        return legal_moves

    def compute_best_move(self, game_state: GameState) -> None:
        self.N = game_state.board.N
        self.rootn = isqrt(self.N)
        self.N2 = self.N*self.N
        # find all legal moves (according to sudoku rules)
        legal_moves = self.find_legal_moves(board=game_state.board)

        def possible(i, j, value, game_state):
            # find only moves for empty squares and non-taboo, legal moves.
            return game_state.board.get(i, j) == SudokuBoard.empty and not\
                TabooMove(i, j, value) in game_state.taboo_moves and \
                value in legal_moves[(i, j)]

        all_moves = [(Move(i, j, value), (i * self.N) + j)
                     for i in range(self.N)
                     for j in range(self.N)
                     for value in range(1, self.N + 1)
                     if possible(i, j, value, game_state)]

        #most_common_numbers = Counter(game_state.board.squares).most_common()
        most_common_numbers = Counter(game_state.board.squares)
        new_moveset = []
        for group in groupby(all_moves, lambda x: x[1]):
            moveset = list(group[1])
            candidate = moveset[0]
            for move in moveset:
                if most_common_numbers[move[0].value]>most_common_numbers[candidate[0].value]:
                    candidate = move
            new_moveset.append(candidate)
        all_moves = new_moveset
        def calcmove(indice, prev_score, calcsquares, calcname):
            nsquares = calcsquares.copy()
            nsquares[indice] = -1
            score = self.countfilled(nsquares)
            scorediff = score - prev_score
            # find children
            indices = [i for i, j in enumerate(nsquares) if j == 0]
            ccountr = 0
            calc_children = []
            for indice in indices:
                cname = calcname + [ccountr]
                ccountr += 1
                calc_children.append({'move': indice, 'name': cname})

            return nsquares, self.scoremap[scorediff], score, calc_children

        initial_score = self.countfilled(game_state.board.squares)

        def get_candidate_node(tag, move):
            name = [f'p{tag}']
            squares, points, score, can_children = calcmove(move[1], initial_score, game_state.board.squares, name)
            return {'name': name, 'move': move[0], 'squares': squares, 'eval': points, 'score': score,
                    'children': can_children}

        tree = [get_candidate_node(nr, move) for nr, move in enumerate(all_moves)]

        # get best candidate for depth=1:
        proposed_move = max(tree, key=lambda x: x['eval']).copy()
        self.propose_move(proposed_move['move'])

        def get_general_node(child):
            child_squares, child_points, child_score, child_children = \
                calcmove(child['move'], parent['score'], parent['squares'], child['name'])

            # the eval is the points lead gained by player, for even moves opponent moves
            # so point gain is negative.
            if len(child['name']) % 2 == 0:
                child_eval = parent['eval'] - child_points
            else:
                child_eval = parent['eval'] + child_points
            as_node = {'name': child['name'], 'eval': child_eval}
            as_move = {'name': child['name'],
                       'squares': child_squares,
                       'score': child_score,
                       'children': child_children,
                       'eval': child_eval}
            return as_node, as_move

        moves = tree
        depth = 0
        while True:
            depth += 1
            proposed_move['eval'] = float('-inf')
            print(f'search depth:{depth}')
            if depth > self.N2:
                break
            for parent in moves:
                if len(parent['children']) > 0:
                    children = [get_general_node(child) for child in parent['children']]
                    child_nodes = list(zip(*children))[0]
                    insertion_pos = next(x for x in tree if x['name'][0] == parent['name'][0])
                    for nesting in parent['name'][1:]:
                        insertion_pos = insertion_pos['eval'][nesting]
                    insertion_pos['eval'] = child_nodes
                    ####################
                    # back propagation #
                    ####################
                    backcopy = deepcopy(tree)

                    def dig(movetree):
                        if type(movetree['eval']) != int:
                            if len(movetree['name']) % 2 == 0:
                                sub = [dig(subtree) for subtree in movetree['eval']]
                                #print(sub)
                                return max(sub)
                            else:
                                sub = [dig(subtree) for subtree in movetree['eval']]
                                #print(sub)
                                return min(sub)
                        else:
                            return movetree['eval']

                    for move in backcopy:
                        move['eval'] = dig(move)
                        if move['eval'] > proposed_move['eval']:
                            proposed_move = move
                            print(f'move:{proposed_move["move"]} and evaluation: {proposed_move["eval"]}')
                            self.propose_move(move['move'])

                    #bestmove = max(backcopy, key=lambda x: x['eval'])
                    #print(f'move:{bestmove["move"]} and evaluation: {bestmove["eval"]}')
                    #self.propose_move(bestmove['move'])
                    # assigning a new depth layer for the moves
                    moves = list(zip(*children))[1]
