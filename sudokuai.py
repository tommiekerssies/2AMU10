from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from operator import itemgetter
from collections import Counter
from itertools import groupby
import operator
import math
import numpy as np


#####################
# for building tree #
#####################
class Node:
    """
    A node somewhere in the tree under the nodes corresponding to possible moves.
    name: a tuple of indices corresponding to all moves played from the current position with the candidate move being
            preceded by a p.
                    example: ('p3',7,5) would mean it's analyzing for a position that occurred after playing on indices
                    3 then 7 then 5.
    squares: a 1D array of the board where hypothetical moves are denoted by -1.
                    example: [0,1,-1,-1] would mean a board that has (0,0) is empty, (0,1) = 1 and the current
                     node analyzes for a path were moves were played in positions: (1,0),(1,1)
    """

    def __init__(self, name, squares, move):
        self.squares = squares
        self.name = name
        self.children = []
        self.move = move
        self.value = 0
        self.visit_count = 0

class AbstractMove:

    def __init__(self, i, j):
        self.i = i
        self.j = j

class CandidateNode:
    """
    Class for the moves right under the root node.
    The same as the Node class only containing an extra parameter move to denote the move that will be played e.g.
    (0,1)->4.
    and an extra parameter static_eval which is the same as eval, but eval updates based on the minimax. However,
    the evaluation of the move on its own is also necessary for some actions.
    """

    def __init__(self, name, squares, move, evaluation):
        self.squares = squares
        self.name = name
        self.children = {}
        self.move = move
        self.eval = evaluation
        self.static_eval = evaluation


###############
# for minimax #
###############

# maps nr. of completed regions to points scored.
scoremap = {0: 0,
            1: 1,
            2: 3,
            3: 7}


class Score:
    """
    This class calculates and keeps track of the score for each node in the tree
    child_score: the total amount of completed regions in the node's child.
    parent_score: the total amount of completed regions in the node.
    sign: whether the moves has a positive or negative effect on the score (if opponents turn, evaluation is negatively
    affected).
    self.score: the evaluation for the current node.
    """

    def __init__(self, child_score, parent_score, sign):
        self.count = parent_score
        self.score = (sign * scoremap[child_score.count - parent_score]) + child_score.score


class Leaf_score:
    """
    The same as the Score class, however since the leaf has no child, the score is 0
    """

    def __init__(self, count):
        self.count = count
        self.score = 0


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # counts the total amount of filled in regions.
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
            for a value located at (i,j). If the value is filled in (>0) remove it from the lists of possible moves
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
        self.n = game_state.board.n
        self.m = game_state.board.m
        # find all legal moves (according to sudoku rules)
        legal_moves = self.find_legal_moves(board=game_state.board)

        def playable(key, value):
            if game_state.board.get(key[0], key[1]) != SudokuBoard.empty:
                return set()
            return set([val for val in value if TabooMove(key[0], key[1], val) not in game_state.taboo_moves])

        def playable_squares(squares, key, value):
            if squares[game_state.board.rc2f(key[0], key[1])] != SudokuBoard.empty:
                return set()
            return set([val for val in value if TabooMove(key[0], key[1], val) not in game_state.taboo_moves])

        playable_moves = {key: playable(key, value) for key, value in legal_moves.items()}
        static_playable_moves = playable_moves.copy()

        passing_exists = False
        for move in playable_moves.keys():

            if len(static_playable_moves[move] - set(playable_moves[move])) != 0:
                passing_exists = True
                passing_move = Move(move[0], move[1], list(static_playable_moves[move] - set(playable_moves[move]))[0])
                print(passing_move)
                break

        def possible(i, j, value, playables):
            return value in playables[(i, j)]

        # find all non forfeiting moves.
        all_moves = [(Move(i, j, value), (i * self.N) + j)
                     for i in range(self.N)
                     for j in range(self.N)
                     for value in range(1, self.N + 1)
                     if possible(i, j, value, playable_moves)]

        def self_play_move(move, squares):
            new_squares = squares.copy()
            index = game_state.board.rc2f(move.i, move.j)
            new_squares[index] = -1
            return new_squares

        def select_node(node, C):
            max_score = 0
            max_child = None
            if not node.children:
                print("Select empty for node: ", node)
            for child in node.children:
                print("Non empty")
                if child.visit_count == 0:
                    print("Zero visit count")
                    return child
                score = child.value / child.visit_count + C * math.sqrt(math.log(node.visit_count) / child.visit_count)
                if score >= max_score:
                    max_score = score
                    max_child = child
            return max_child

        def get_moveset(node):
            # find all non forfeiting moves.
            all_moves = [AbstractMove(i, j)
                         for i in range(self.N)
                         for j in range(self.N)
                         if node.squares[game_state.board.rc2f(i, j)] == SudokuBoard.empty]
            return all_moves

        def exploit_tree_knowledge(start_node, epsilon, movetree, turn, score):
            child = start_node
            complete_moveset = get_moveset(start_node)
            if len(complete_moveset) == 0:
                won = score > 0
                return movetree, won

            if start_node.children == [] or epsilon > np.random.uniform():
                chosen_move = np.random.choice(len(complete_moveset))
                child = Node("", self_play_move(complete_moveset[chosen_move], child.squares),
                             Move(complete_moveset[chosen_move].i, complete_moveset[chosen_move].j, -1))
                print("Random")
            else:
                print("Greedy")
                child = max(start_node.children, key=lambda k: k.value/k.visit_count)

            if turn == 0:
                score += self.countfilled(child.squares) - self.countfilled(start_node.squares)
                turn = 1
            else:
                score -= self.countfilled(child.squares) - self.countfilled(start_node.squares)
                turn = 0

            movetree.append(child)
            return exploit_tree_knowledge(child, epsilon, movetree, turn, score)

        game_tree = self.load()
        if game_tree is None:
            game_tree = Node("root", game_state.board.squares.copy(), None)
            game_tree.visit_count = 1
            game_tree.children = [Node(f'{move[0].i},{move[0].j},{move[0].value}',
                                       self_play_move(move[0], game_tree.squares), move[0]) for move in all_moves]
        else:
            for child in game_tree.children:
                if child.move.value == -1:
                    child_index = game_state.board.rc2f(child.move.i, child.move.j)
                    valid_moves = [move.value for move, k in all_moves if k == child_index]
                    for move in valid_moves:
                        new_move = Move(child.move.i, child.move.j, move)
                        child_v = Node(child.name, child.squares, new_move)
                        child_v.value = child.value
                        child_v.visit_count = child.visit_count
                        child_v.children = child.children
                        game_tree.children.append(child_v)


            game_tree.children[:] = [child for child in game_tree.children if
                                     child.move.value in [move.value for move, k in all_moves
                                                          if k == game_state.board.rc2f(child.move.i, child.move.j)]]

        def append_node(move, root, path):
            node = root
            print("Appending move ", move.move)
            idx1 = 0
            while idx1+1 < len(path):
                for child in node.children:
                    if child.squares == path[idx1+1].squares:
                        node = child
                        idx1 += 1
                        break
            node.children.append(move)

        # MCTS
        C = 0.4
        while True:
            selected_node = select_node(game_tree, C)
            print("Simulation number", selected_node.visit_count)
            move_tree, won = exploit_tree_knowledge(selected_node, 0.05, [selected_node], 0, 0)
            increment = 0
            expanded = False
            if won:
                increment += 1
            for move in move_tree:
                print("Move: ", move.squares)

                # Expand tree by one node every simulation
                if(move.visit_count == 0 and not expanded):
                    depth = move_tree.index(move)
                    append_node(move, selected_node, move_tree[:depth])
                    expanded = True

                # Update node statistics
                move.visit_count += 1
                move.value += increment


            # Propose and save after each simulation

            chosen_move = max(game_tree.children, key=lambda k:
                k.value/k.visit_count if k.visit_count > 0 else 0)

            self.propose_move(chosen_move.move)
            #self.save(game_tree.children[game_tree.children.index(chosen_move)])


            # Backprop

        # Old stuff

