import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from math import isqrt, floor
from operator import itemgetter
from copy import deepcopy

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

        # initialize parameters
        #board size

        #list of possible moves
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

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        self.N = game_state.board.N
        self.rootn = isqrt(self.N)
        self.N2 = self.N*self.N
        # find all legal moves (according to sudoku rules)
        legal_moves = self.find_legal_moves(board=game_state.board)

        def possible( i, j, value, game_state):
            # find only moves for empty squares and non-taboo, legal moves.
            return game_state.board.get(i, j) == SudokuBoard.empty and not\
                TabooMove(i, j, value) in game_state.taboo_moves and \
                value in legal_moves[(i, j)]





        all_moves = [(Move(i, j, value), (i*self.N)+j )
                     for i in range(self.N)
                     for j in range(self.N)
                     for value in range(1, self.N+1)
                     if possible(i, j, value, game_state)]

        def calcmove(indice, prev_score, calcsquares, calcname, parent_move):
            nsquares = calcsquares.copy()
            nsquares[indice] = -1
            score = self.countfilled(nsquares)
            scorediff = score - prev_score

            #find children
            indices = [i for i, j in enumerate(nsquares) if j == 0]
            ccountr = 0
            children = []
            for indice in indices:
                cname = calcname +[ccountr]
                ccountr+=1
                children.append({'parent': calcname, 'parent_move': parent_move, 'move': indice, 'name': cname})

            return nsquares, self.scoremap[scorediff], score, children

        initial_score = self.countfilled(game_state.board.squares)
        tree = []
        countr = 0
        for move in all_moves:
            name = [f'p{countr}']
            countr+=1
            squares, points, score, children = calcmove(move[1], initial_score, game_state.board.squares, name, move[0])
            tree.append({'name': name,
                         'move': move[0],
                         'squares': squares,
                         'points': points,
                         'tally': points,
                         'score': score,
                         'children': children})


        maxp = -1
        for package in tree:
            if package['points']>maxp:
                pmove = package['move']

        self.propose_move(pmove)

        moves = tree

        tree = []
        for parent in moves:
            tree.append({'move': parent['move'], 'name': parent['name'], 'eval': [parent['tally']]})
        maximize = True
        depth = 0
        while True:
            depth +=1
            print(f'search depth:{depth}')
            allchildren = []
            maximize = not maximize
            for parent in moves:
                group_children = []
                if len(parent['children']) > 0:
                    childcount = 0
                    for child in parent['children']:
                        child_squares, child_points, child_score, child_children = \
                            calcmove(child['move'], parent['score'], parent['squares'], child['name'], parent['move'])
                        if maximize:
                            child_tally = parent['tally'] + child_points
                        else:
                            child_tally = parent['tally'] - child_points
                        child_name = parent['name']+[childcount]
                        allchildren.append({'children': child_children,
                                            'name': child_name,
                                            'score': child_score,
                                            'squares': child_squares,
                                            'move': parent['move'],
                                            'tally': child_tally}.copy())
                        group_children.append({'name': child_name, 'eval': child_tally})
                        childcount += 1


                    rootname = next(x for x in tree if x['name'][0] == parent['name'][0])
                    treeslice = rootname
                    for pos in parent['name'][1:]:
                        treeslice = treeslice['eval'][pos]
                    treeslice['eval'] = group_children
            # backprop
            def kfunc_cmove(k):
                return k['eval']
            backcopy = deepcopy(tree)


            def dig(movetree, maxim):
                if type(movetree) == int:
                    return movetree
                elif type(movetree['eval']) != int:
                    if maxim:
                        return max([dig(subtree, not maxim) for subtree in movetree['eval']])
                    else:
                        return min([dig(subtree, not maxim) for subtree in movetree['eval']])
                else:
                    return movetree['eval']

            for move in backcopy:
                maximize = True
                move['eval'] = dig(move, not maximize)

            bestmove = max(backcopy, key=kfunc_cmove)['move']
            print(f'proposing:{bestmove}')
            self.propose_move(bestmove)

            moves = allchildren








