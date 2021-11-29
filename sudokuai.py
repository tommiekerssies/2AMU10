#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty and not TabooMove(i, j, value) in game_state.taboo_moves

        all_moves = [Move(i, j, value) for i in range(N) for j in range(N) for value in range(1, N+1) if possible(i, j, value)]
        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))

def get_move_score(move: Move, board: SudokuBoard) -> int:
    new_regions_completed = 0
    board.put(move.i, move.j, move.value)

    # check if a column is completed
    for j in range(board.N):
        if board.get(j, move.i) is None:
            new_regions_completed += 1
            break

    # check if a row is completed
    for i in range(board.N):
        if board.get(move.j, i) is None:
            new_regions_completed += 1
            break

    # check if a block is completed
    # TODO!

    score = 0
    if new_regions_completed is 0:
        score = 0
    if new_regions_completed is 1:
        score = 1
    if new_regions_completed is 2:
        score = 3
    if new_regions_completed is 3:
        score = 7

    return score
