#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from termcolor import colored
import random
from colorama import Fore, Style


def create_board(rows, cols):
    return np.zeros((rows, cols))


def is_valid_move(board, col, rows):
    if col < 0 or col >= board.shape[1]:
        return False
    return np.all(board[rows - 1, col] == 0.0)


def make_move(board, col, player, rows):
    for r in range(rows):
        if board[r, col] == 0:
            board[r, col] = player
            break
    return board


def is_board_full(board, rows, cols):
    return np.all(board[rows - 1, :] != 0.0)


def has_valid_moves(board, rows, cols):
    for col in range(cols):
        if is_valid_move(board, col, rows):
            return True
    return False


def print_board(board, rows, cols):
    for r in range(rows - 1, -1, -1):
        row_str = ""
        for c in range(cols):
            if board[r, c] == 1:
                row_str += "\033[91mX\033[0m "
            elif board[r, c] == 2:
                row_str += "\033[94mO\033[0m "
            else:
                row_str += ". "
        print(row_str)
    print("-" * (2 * cols - 1))
    print(" ".join(str(i) for i in range(cols)))


def evaluate(board, rows, cols, length):
    AI_PLAYER = 2
    HUMAN_PLAYER = 1

    def score_window(window, player, length):
        score = 0
        opponent = HUMAN_PLAYER if player == AI_PLAYER else AI_PLAYER
        opponent_count = np.count_nonzero(window == opponent)

        if opponent_count == 0:
            num_player_pieces = np.count_nonzero(window == player)
            gap_sequences = ["X.XX", "XX.X", ".XXX",
                             "XXX.", "X.X.X", "X..XX", "XX..X"]
            gap_sequence_found = False

            def count_gap_sequences(arr):
                count = 0
                arr_str = "".join(["X" if x == player else "." for x in arr])

                for seq in gap_sequences:
                    if seq in arr_str:
                        count += 1

                return count

            gap_sequence_found = count_gap_sequences(window)

            if num_player_pieces == length - 1:
                score += 10000 if player == AI_PLAYER else -20000
            elif num_player_pieces == length - 2 or gap_sequence_found:
                multiplier = 1 if gap_sequence_found else num_player_pieces ** 2
                score += 2000 * multiplier if player == AI_PLAYER else -500 * multiplier
            elif num_player_pieces == length - 3:
                score += 500 * \
                    (num_player_pieces ** 2) if player == AI_PLAYER else - \
                    100 * (num_player_pieces ** 2)

        return score

    windows = []

    # Horizontal windows
    for r in range(rows):
        for c in range(cols - length + 1):
            windows.append(board[r, c:c + length])

    # Vertical windows
    for r in range(rows - length + 1):
        for c in range(cols):
            windows.append(board[r:r + length, c])

    # Positive diagonal windows
    for r in range(rows - length + 1):
        for c in range(cols - length + 1):
            windows.append(np.diagonal(board[r:r + length, c:c + length]))

    # Negative diagonal windows
    for r in range(length - 1, rows):
        for c in range(cols - length + 1):
            windows.append(np.diagonal(np.fliplr(board)[
                           r - length + 1:r + 1, c:c + length]))

    ai_score = 0
    human_score = 0

    for window in windows:
        ai_score += score_window(window, AI_PLAYER, length)
        human_score += score_window(window, HUMAN_PLAYER, length)

    result = ai_score - human_score

    return result


def is_winning_move(board, player, rows, cols, length):
    # Check horizontal locations for a win
    for r in range(rows):
        for c in range(cols - length + 1):
            horizontal = np.all(board[r, c:c + length] == player)

            if horizontal:
                return True

    # Check vertical locations for a win
    for r in range(rows - length + 1):
        for c in range(cols):
            vertical = np.all(board[r:r + length, c] == player)

            if vertical:
                return True

    # Check positively sloped diagonals
    for r in range(rows - length + 1):
        for c in range(cols - length + 1):
            pos_diagonal = np.all(
                np.array([board[r + i, c + i] for i in range(length)]) == player)

            if pos_diagonal:
                return True

    # Check negatively sloped diagonals
    for r in range(length - 1, rows):
        for c in range(cols - length + 1):
            neg_diagonal = np.all(
                np.array([board[r - i, c + i] for i in range(length)]) == player)

            if neg_diagonal:
                return True

    return False


def minimax(board, depth, maximizing_player, rows, cols, length):
    if depth == 0 or is_winning_move(board, 1, rows, cols, length) or is_winning_move(board, 2, rows, cols, length):
        return evaluate(board, rows, cols, length), None

    if maximizing_player:
        value = float('-inf')
        col = None
        for c in range(cols):
            if is_valid_move(board, c, rows):
                new_board = make_move(np.copy(board), c, 1, rows)
                last_move_row = np.argmax(new_board[:, c] != 0) - 1
                if is_winning_move(new_board, 1, rows, cols, length):
                    return evaluate(new_board, rows, cols, length), c
                new_value, _ = minimax(
                    new_board, depth - 1, False, rows, cols, length)
                if new_value > value:
                    value = new_value
                    col = c
        return value, col
    else:
        value = float('inf')
        col = None
        for c in range(cols):
            if is_valid_move(board, c, rows):
                new_board = make_move(np.copy(board), c, 2, rows)
                last_move_row = np.argmax(new_board[:, c] != 0) - 1
                if is_winning_move(new_board, 2, rows, cols, length):
                    return evaluate(new_board, rows, cols, length), c
                new_value, _ = minimax(
                    new_board, depth - 1, True, rows, cols, length)
                if new_value < value:
                    value = new_value
                    col = c
        return value, col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=6,
                        help='Number of rows in the board')
    parser.add_argument('--cols', type=int, default=7,
                        help='Number of columns in the board')
    parser.add_argument('--length', type=int, default=4,
                        help='Length of consecutive pieces to win')
    parser.add_argument('--depth', type=int, default=4,
                        help='Depth of the minimax algorithm')
    args = parser.parse_args()

    board = create_board(args.rows, args.cols)
    print_board(board, args.rows, args.cols)  # Display the initial empty board
    game_over = False

    HUMAN_PLAYER = 1
    AI_PLAYER = 2

    turn = random.choice([HUMAN_PLAYER, AI_PLAYER])

    while not game_over:
        if turn == HUMAN_PLAYER:
            # Human player's move
            col = None
            while col is None:
                try:
                    col_input = input(f"Enter your move (0-{args.cols - 1}): ")
                    if col_input == "":
                        print("Please enter a valid column number.")
                    else:
                        col = int(col_input)
                        if not is_valid_move(board, col, args.rows):
                            print("Invalid move. Please try again.")
                            col = None
                except ValueError:
                    print("Invalid input. Please enter a valid column number.")
            if not is_valid_move(board, col, args.rows):
                print("Invalid move. Try again.")
                continue
            board = make_move(board, col, HUMAN_PLAYER, args.rows)
            # board = make_move(board, col, 1, args.rows)
            # Display the initial empty board
            print_board(board, args.rows, args.cols)
            if is_winning_move(board, 1, args.rows, args.cols, args.length):
                print("You win!")
                game_over = True
            turn = AI_PLAYER

        else:
            # AI's move
            _, col = minimax(board, args.depth, True,
                             args.rows, args.cols, args.length)
            board = make_move(board, col, AI_PLAYER, args.rows)
            # board = make_move(board, col, 2, args.rows)
            print(f"AI's move: {col}")
            # Display the initial empty board
            print_board(board, args.rows, args.cols)
            if is_winning_move(board, 2, args.rows, args.cols, args.length):
                print("AI wins!")
                game_over = True

            if not has_valid_moves(board, args.rows, args.cols) and not game_over:
                print("It's a draw!")
                game_over = True
            turn = HUMAN_PLAYER


if __name__ == '__main__':
    main()
