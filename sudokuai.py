from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from operator import itemgetter
from itertools import groupby
from collections import Counter
import operator


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
    def __init__(self, name, squares):
        self.squares = squares
        self.name = name
        self.children = {}
        self.eval = None


class CandidateNode:
    """
    Class for the moves right under the root node.
    The same as the Node class only containing an extra parameter move to denote the move that will be played e.g.
    (0,1)->4.
    and an extra parameter static_eval which is the same as eval, but eval updates based on the minimax. However,
    the evaluation of the move on its own is also necessary for some actions.
    """
    def __init__(self, name, squares, move, evaluation, static_evaluation):
        self.squares = squares
        self.name = name
        self.children = {}
        self.move = move
        self.eval = evaluation
        self.static_eval = static_evaluation


###############
# for minimax #
###############

# maps nr. of completed regions to points scored.
scoremap = {-3: -7,
            -2: -3,
            -1: -1,
            0: 0,
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
        self.score = (sign * scoremap[child_score.count-parent_score])+child_score.score


class Leaf_score:
    """
    The same as the Score class, however since the leaf has no child, the score is 0
    """
    def __init__(self, count, namelen):
        # amount of filled regions
        self.count = count[0]
        # the points the opponent could score if they were to move
        if namelen % 2 == 0:
            self.score = scoremap[count[1]]
        else:
            self.score = -scoremap[count[1]]


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def get_splits(self, squares):
        # split the data into groups representing rows and columns
        row_split = [squares[x:x + self.N] for x in range(0, len(squares), self.N)]
        col_split = [itemgetter(*[(self.N * i) + j for i in range(self.N)])(squares) for j in range(self.N)]
        block_split = [itemgetter(*index)(squares) for index in self.split_indices_block]
        return row_split, col_split, block_split

    def get_most_common_eight(self, row_split, col_split, block_split):
        # Find the positions of the missing values in the 8-full rows/cols/blocks and add the indexes to a list
        where_eights_occur = [self.split_indices_block[blocknr][block.index(0)]
                              for blocknr, block in enumerate(block_split) if block.count(0) == 1] + \
                             [rownr * self.N + row.index(0)
                              for rownr, row in enumerate(row_split) if row.count(0) == 1] + \
                             [colnr + col.index(0) * self.N
                              for colnr, col in enumerate(col_split) if col.count(0) == 1]
        # Find the most occuring index (would be 3 if one missing value can complete a row, col and block)
        most_common = Counter(where_eights_occur).most_common(1)
        if not most_common:
            return None
        return Counter(where_eights_occur).most_common(1)[0]

    # counts the total amount of filled in regions.
    def countfilled(self, squares,leaf):
        """
        @param squares: the squares parameter from the game state
        @return: the total amount of filled columns, rows and blocks
        The function takes an array of length N*N and computes, when converted to a sudoku board, how many rows, columns
        and blocks don't contain any filled-in values. it returns the sum of the amount of rows, columns and blocks.
        """
        row_split, col_split, block_split = self.get_splits(squares)

        # count all the empty values for each group in the rows, cols and blocks, then take the total of the three.
        nrfilled = sum([[sublist.count(0) for sublist in zone].count(0)
                        for zone in [row_split, col_split, block_split]])
        if not leaf:
            return nrfilled

        # Thinking one ply ahead by seeing if the opponent can score and how much given the board
        # Only useful for leaf nodes because in other situations we already analyze what will happen next
        negscore = 0

        # count the amount of rows/cols/blocks with exactly 8 values filled in
        row_contains_eight = [sublist.count(0) for sublist in row_split].count(1)
        col_contains_eight = [sublist.count(0) for sublist in col_split].count(1)
        block_contains_eight = [sublist.count(0) for sublist in block_split].count(1)
        total_eights = row_contains_eight + col_contains_eight + block_contains_eight
        list_total_eights = [row_contains_eight, col_contains_eight, block_contains_eight]

        # find how many points the opponent can score on its next turn based on the position of the 8-fulls
        if total_eights > 0:
            # if all the 8-fulls are in only the rows,cols or blocks, each index can only occur once so negscore = 1
            if total_eights not in list_total_eights:
                most_common_eight = self.get_most_common_eight(row_split, col_split, block_split)
                negscore = most_common_eight[1]
            else:
                negscore = 1

        return nrfilled, negscore

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

        # split the data into groups representing blocks
        indices_block = [self.N * (j // self.n) + j % self.n + (i * self.n) % self.N + (i // self.m) * self.N * self.m
                         for i in range(self.N) for j in range(self.N)]
        self.split_indices_block = [indices_block[x:x + self.N] for x in range(0, len(indices_block), self.N)]

        # find all legal moves (according to sudoku rules)
        legal_moves = self.find_legal_moves(board=game_state.board)

        def find_playable_moves(key, value):
            if game_state.board.get(key[0], key[1]) != SudokuBoard.empty:
                return set()
            return set([val for val in value if TabooMove(key[0], key[1], val) not in game_state.taboo_moves])
        playable_moves = {key: find_playable_moves(key, value) for key, value in legal_moves.items()}
        static_playable_moves = playable_moves.copy()

        ###############
        # Heuristics: #
        ###############

        # for the i'th block, find all combinations of (i,j) that fall in that block
        block_lookup_table = {i: [] for i in range(self.N)}
        [block_lookup_table[self.m * (i // self.m) + j // self.n].append((i, j))
         for i in range(self.N) for j in range(self.N)]

        ###########################
        # find all hidden singles #
        ###########################
        # for all combinations of (i,j) see if there exist a possible move which is not allowed in any other square
        # in the row, column or block of (i,j). Then make this move the only legal move for position (i,j)
        for i in range(self.N):
            for j in range(self.N):
                current_move = playable_moves[(i, j)]
                if len(current_move) > 1:
                    # check for row, column, block if there is a number which cannot be filled in anywhere but on (i,j)
                    row_moves = [playable_moves[(i, row)] for row in range(self.N) if row != j]
                    row_hsingle = set(current_move) - set.union(*row_moves)
                    if len(row_hsingle) == 1:
                        playable_moves[(i, j)] = row_hsingle
                        continue

                    col_moves = [playable_moves[(col, j)] for col in range(self.N) if col != i]
                    col_hsingle = set(current_move) - set.union(*col_moves)
                    if len(col_hsingle) == 1:
                        playable_moves[(i, j)] = col_hsingle
                        continue

                    block_moves = [playable_moves[pos] for pos in
                                   block_lookup_table[self.m * (i // self.m) + j // self.n] if pos != (i, j)]
                    block_hsingle = set(current_move) - set.union(*block_moves)
                    if len(block_hsingle) == 1:
                        playable_moves[(i, j)] = block_hsingle
                        continue

        ###################################
        # execute subgroup-exclusion heuristic #
        ###################################

        # Check if all indices in a bin are in the same block
        def same_block(bin):
            blocks = [(bin_i // self.N // self.n) * self.n + (bin_i % self.N) // self.m for bin_i in bin]
            if len(set(blocks)) == 1:
                # print("Found in the same block for bin ", bin)
                return True
            else:
                return False

        # Obtain the list of legal moves for the block that square bin_i is part of
        def get_block_moves(bin_i):
            block = (bin_i // self.N // self.n) * self.n + (bin_i % self.N) // self.m
            first_in_block = game_state.board.f2rc((block // self.n) * self.N * self.n + (block % self.n) * self.m)
            moves = {}
            for i in range(self.n):
                for k in range(self.m):
                    moves[(first_in_block[0] + i, first_in_block[1] + k)] = (
                        playable_moves[(first_in_block[0] + i, first_in_block[1] + k)])
            return moves

        # Execute subgroup-exclusion heuristic for the columns
        for i in range(self.N):
            row_moves = [playable_moves[(row, i)] for row in range(self.N)]
            if len(row_moves) > 1:
                v_bins = []
                for v in range(1, self.N + 1):
                    v_bin = [(i + index * self.N) for index, row in enumerate(row_moves) if v in row
                             and game_state.board.get(index, i) == SudokuBoard.empty]
                    v_bins.append(v_bin)
                for bin_val, bin in enumerate(v_bins):
                    if (same_block(bin)):
                        """ Cross out posibilities """
                        cross_moves = get_block_moves(bin[0])
                        actual_to_cross = [ind for ind in cross_moves if (game_state.board.rc2f(ind[0], ind[1]))
                                           not in bin and game_state.board.get(ind[0], ind[1]) == SudokuBoard.empty]
                        for index in actual_to_cross:
                            inter = set(playable_moves[index])
                            inter.discard(bin_val + 1)
                            legal_moves[index] = inter

        # Execute subgroup-exclusion heuristic for the rows
        for i in range(self.N):
            row_moves = [playable_moves[(i, row)] for row in range(self.N)]
            if len(row_moves) > 1:
                v_bins = []
                for v in range(1, self.N + 1):
                    v_bin = [(i * self.N + index) for index, row in enumerate(row_moves) if v in row
                             and game_state.board.get(i, index) == SudokuBoard.empty]
                    v_bins.append(v_bin)
                for bin_val, bin in enumerate(v_bins):
                    # print("Checking for bin ", bin_val + 1)
                    if (same_block(bin)):
                        """ Cross out posibilities """
                        cross_moves = get_block_moves(bin[0])
                        actual_to_cross = [ind for ind in cross_moves if
                                           (game_state.board.rc2f(ind[0], ind[1]))
                                           not in bin and game_state.board.get(ind[0], ind[
                                               1]) == SudokuBoard.empty]
                        for index in actual_to_cross:
                            inter = set(playable_moves[index])
                            inter.discard(bin_val + 1)
                            legal_moves[index] = inter

        ###################################
        # find all hidden tuples in a row #
        ###################################
        # for all rows...
        for i in range(self.N):
            # get the playable moves per cell
            row_moves = [playable_moves[(i, row)] for row in range(self.N)]
            # if there are less than two cells in the row there cannot be a tuple
            if len(row_moves) <= 2:
                continue
            # for all cells in the row...
            for j1, moves1 in enumerate(row_moves):
                # we are trying to reduce the possible moves in this cell to 2 so if it is already 2
                # or less possible moves we can skip it
                if len(moves1) <= 2:
                    continue
                # go through the cells again to find a potential twin
                for j2, moves2 in enumerate(row_moves):
                    # skip the cell combi's we already covered
                    if j2 <= j1:
                        continue
                    # make a candidate set of moves for this potential hidden tuple
                    htuple_candidate = list(set(moves1) & set(moves2))
                    # now keep checking whether the candidate set is indeed a hidden tuple
                    while len(htuple_candidate) > 1:
                        # assume it is a hidden tuple until proven otherwise
                        is_htuple = True
                        # now go through the cells again for a third time, because we want to find a counterexample
                        for j3, moves3 in enumerate(row_moves):
                            # skip the cells for the potential hidden tuple itself
                            if j3 == j1 or j3 == j2:
                                continue
                            # check if there is a move in this cell that is also in the candidate set, and if it is
                            # remove it from the candidate set and we will try again if the candidate set is still
                            # bigger than one in length
                            for move in htuple_candidate:
                                if move in moves3:
                                    is_htuple = False
                                    htuple_candidate.remove(move)
                                    break
                            if not is_htuple:
                                break
                        # if it turns out to be a hidden tuple, update the possible moves by the candidate set
                        if is_htuple:
                            playable_moves[(i, j1)] = htuple_candidate
                            playable_moves[(i, j2)] = htuple_candidate
                            break

        ######################################
        # find all hidden tuples in a column #
        ######################################
        # almost same logic as in a row, see the comments there
        for j in range(self.N):
            col_moves = [playable_moves[(col, j)] for col in range(self.N)]
            if len(col_moves) <= 2:
                continue
            for i1, moves1 in enumerate(col_moves):
                if len(moves1) <= 2:
                    continue
                for i2, moves2 in enumerate(col_moves):
                    if i2 <= i1:
                        continue
                    htuple_candidate = list(set(moves1) & set(moves2))
                    while len(htuple_candidate) > 1:
                        is_htuple = True
                        for i3, moves3 in enumerate(col_moves):
                            if i3 == i1 or i3 == i2:
                                continue
                            for move in htuple_candidate:
                                if move in moves3:
                                    is_htuple = False
                                    htuple_candidate.remove(move)
                                    break
                            if not is_htuple:
                                break
                        if is_htuple:
                            playable_moves[(i1, j)] = htuple_candidate
                            playable_moves[(i2, j)] = htuple_candidate
                            break

        #####################################
        # find all hidden tuples in a block #
        #####################################
        # almost same logic as in a row, see the comments there
        for b in range(len(block_lookup_table)):
            block_moves = [(pos, playable_moves[pos]) for pos in block_lookup_table[b]]
            if len(block_moves) <= 2:
                continue
            for b_i1, moves1 in enumerate(block_moves):
                if len(moves1) <= 2:
                    continue
                for b_i2, moves2 in enumerate(block_moves):
                    if b_i2 <= b_i1:
                        continue
                    htuple_candidate = list(set(moves1) & set(moves2))
                    while len(htuple_candidate) > 1:
                        is_htuple = True
                        for b_i3, (_, moves3) in enumerate(block_moves):
                            if b_i3 == b_i1 or b_i3 == b_i2:
                                continue
                            for move in htuple_candidate:
                                if move in moves3:
                                    is_htuple = False
                                    htuple_candidate.remove(move)
                                    break
                            if not is_htuple:
                                break
                        if is_htuple:
                            playable_moves[block_moves[b_i1][0]] = htuple_candidate
                            playable_moves[block_moves[b_i2][0]] = htuple_candidate
                            break

        #####################
        # Find passing move #
        #####################
        # find the first move that was pruned out by the heuristics and denote it as a passing move
        # playing this move will result in the sudoku having no solutions allowing the agent to pass a turn
        passing_exists = False
        for move in playable_moves.keys():
            if len(static_playable_moves[move] - set(playable_moves[move])) != 0:
                passing_exists = True
                passing_move = Move(move[0],move[1], list(static_playable_moves[move] - set(playable_moves[move]))[0])
                print(passing_move)
                break

        # check for each move if it can be played according to the heuristic
        def possible(i, j, value):
            return value in playable_moves[(i, j)]

        # find all non forfeiting moves.
        # pruned by the heuristics
        all_moves = [(Move(i, j, value), (i * self.N) + j)
                     for i in range(self.N)
                     for j in range(self.N)
                     for value in range(1, self.N + 1)
                     if possible(i, j, value)]

        ############################
        # Assign most common moves #
        ############################

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

        #####################
        # Evaluate greedily #
        #####################

        # calculate various additional parameters for each move
        def calcmove(indice, calcsquares):
            """
            @param indice:  where on the sudoku board the move will take place
            @param calcsquares: a 1D array of the board before prior to the move being played
            @param calcname: the name of the parent of this node
            @return:
                nsquares: a 1D array of the board after the move has been played
                scorediff: the amount of points scored by playing the move
                calc_children: a list of the moves that can be played after playing this move
            Calculates the state of the board and score after playing a hypothethical move.
            """
            prev_score = self.countfilled(calcsquares, leaf=False)
            nsquares = calcsquares.copy()
            # -1 is used to denote a move has been played on a square without needing to specify what the number is.
            nsquares[indice] = -1
            score, negative_score = self.countfilled(nsquares, leaf=True)
            static_scorediff = score - prev_score
            scorediff = static_scorediff - negative_score
            return nsquares, scoremap[scorediff], static_scorediff

        def get_candidate_node(tag, move):
            name = (f'p{tag}',)
            squares, points, static_points = calcmove(move[1], game_state.board.squares)
            if points == 0:
                return None
            return CandidateNode(name, squares, move[0], points, static_points)

        # construct a tree as a collection of the direct children of the root node (candidate nodes)
        tree = {}
        for nr, move in enumerate(all_moves):
            candidate_node = get_candidate_node(nr, move)
            if candidate_node:
                tree[(f'p{nr}',)] = candidate_node

        def propose_most_common_eight_else_random_move():
            row_split, col_split, block_split = self.get_splits(game_state.board.squares)
            most_common_eight = self.get_most_common_eight(row_split, col_split, block_split)
            if most_common_eight:
                most_common_eight_k = most_common_eight[0]
                i, j = game_state.board.f2rc(most_common_eight_k)
                value = list(playable_moves[(i, j)])[0]
                move = Move(i, j, value)
                self.propose_move(move)
            else:
                for (i,j), random_move_set in playable_moves.items():
                    random_moves = list(random_move_set)
                    if len(random_moves) > 0:
                        self.propose_move(Move(i, j, random_moves[0]))

        # propose the best candidate node.
        if len(tree) == 0:
            if passing_exists:
                self.propose_move(passing_move)
            else:
                propose_most_common_eight_else_random_move()
        else:
            pmove = max(list(tree.values()), key=operator.attrgetter('eval')).move
            print(f'd0, proposed move is {pmove}')
            self.propose_move(pmove)

        ######################
        # Minimax evaluation #
        ######################

        def find_children(parent, greedy):
            # get the children of a node, with name denothing what the path of the node is, and squares the occupied
            # positions on the board.
            children = {}
            for i, value in enumerate(parent.squares):
                if value != 0:
                    continue
                squares, points, static_points = calcmove(i, parent.squares)
                if points == 0 and greedy:
                    continue
                children[parent.name+(i,)] = Node(parent.name+(i,), squares)
            return children

        def delve(list_tree):
            # get all children of the list_tree
            list_tree = [o.children for o in list_tree]
            list_tree = {key: value for sublist in list_tree for key, value in zip(sublist.keys(),sublist.values())}
            return list_tree

        depth = 0
        while True:
            #################
            # Building Tree #
            #################

            # get the set of leaf nodes
            moves = tree
            for i in range(depth):
                moves = delve(moves.values())

            # if the board is full stop running
            if len(moves) == 0:
                print('killed')
                break

            # find the children of the leaf nodes and add them to the tree
            for parent in moves.values():
                parent.children = find_children(parent, greedy=depth % 2 != 0)

            # increment the depth
            depth += 1

            #######################################
            # Performing Minimax with a/b pruning #
            #######################################

            def minimax(movetree, alpha, beta):
                """
                Minimax with a/b pruning.
                Calculates the evaluation for the root node in movetree.
                @param movetree: a tree or subtree of moves with a single root node.
                @param alpha: minimax parameter
                @param beta:  minimax parameter
                @return: evaluation for the root node.
                """

                # score the leaves
                if len(movetree.children) == 0:
                    return Leaf_score(self.countfilled(movetree.squares, leaf=True), len(movetree.name))
                # score the nodes
                else:
                    current_score = self.countfilled(movetree.squares, leaf=False)
                    # name indicates path so its polarity shows whether it is a min or max layer.
                    if len(movetree.name) % 2 == 0:
                        max_node = None
                        max_eval = float('-inf')
                        # get the most favorable evaluation of the children
                        for child_tree in movetree.children.values():
                            # compute the evaluation of the child using minimax
                            curr_eval = Score(minimax(child_tree, alpha, beta), current_score, 1)
                            child_tree.eval = curr_eval.score
                            if max_eval < curr_eval.score:
                                max_node = curr_eval
                                max_eval = curr_eval.score
                            alpha = max(alpha, curr_eval.score)
                            if beta <= alpha:
                                break
                        return max_node

                    else:
                        # same procedure as above but for min layers.
                        min_node = None
                        min_eval = float('inf')
                        for child_tree in movetree.children.values():
                            curr_eval = Score(minimax(child_tree, alpha, beta), current_score, -1)
                            child_tree.eval = curr_eval.score
                            if min_eval > curr_eval.score:
                                min_node = curr_eval
                                min_eval = curr_eval.score
                            beta = min(beta, curr_eval.score)
                            if beta <= alpha:
                                break
                        return min_node

            # evaluate all candidate nodes and propose the strongest.
            """This generally isn't the slowest step but things can go wrong if the program is forced to halt here
            as it proposes moves iteratively, so it can propose the first move, which may be garbage and then not have
            time to propose a better move."""
            for move in tree.values():
                # uses the static eval because otherwise things go wrong when looking at deeper levels.
                move.eval = move.static_eval + minimax(move, float('-inf'), float('inf')).score

            if len(tree) == 0:
                if passing_exists:
                    self.propose_move(passing_move)
                else:
                    propose_most_common_eight_else_random_move()
            else:
                pmove = max(list(tree.values()), key=operator.attrgetter('eval'))
                if pmove.eval < 0 and passing_exists:
                    self.propose_move(passing_move)
                    print('trying to pass')
                elif pmove.eval == 0 and passing_exists and game_state.board.squares.count(0) % 2 == 1:
                    self.propose_move(passing_move)
                    print('passing to get last move')
                else:
                    self.propose_move(pmove.move)
                    print(f'proposing: {pmove.move}\n with evaluation:{pmove.eval}')
                    print(f'finished depth {depth}')

            ##################################################
            # Sorting Tree to increase pruning on next depth #
            ##################################################

            def order(branch):
                """
                Orders a minimax tree based on evaluations.
                @param branch: The children of some node in the tree e.g. all the candidate nodes
                                (children of root node)
                @return: an ordered version of the branch
                """
                if len(branch) == 0:
                    return branch
                # if the nodes have children, also order the children.
                if len(list(branch.values())[0].children) != 0:
                    for branch_move in branch.values():
                        branch_move.children = order(branch_move.children)
                # based on whether it is a min or max layer (represented by the parity of a node's name) sort the nodes.
                """None values appear where the tree was pruned during minimax, these branches should keep being pruned
                as long as they appear after all the nodes that came before them, therefore all pruned nodes are
                placed at the back."""
                if len(list(branch.keys())[0]) % 2 == 0:
                    ordered_branch = sorted(branch.items(),
                                            key=lambda x: float('inf') if not x[1].eval else x[1].eval, reverse=False)
                else:
                    ordered_branch = sorted(branch.items(),
                                            key=lambda x: float('-inf') if not x[1].eval else x[1].eval, reverse=True)

                return {move_name_combo[0]: move_name_combo[1] for move_name_combo in ordered_branch}

            # orders the current tree based on evaluations.
            tree = order(tree)
