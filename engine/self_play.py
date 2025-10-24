import chess
import random
import torch
from board import encode_board

def play_random_game(max_moves=50):
    board = chess.Board()
    states = []

    while not board.is_game_over() and len(states) < max_moves:
        encoded = encode_board(board)
        states.append(encoded)

        # Pick a random legal move
        move = random.choice(list(board.legal_moves))
        board.push(move)

    # Game result: +1 (white win), 0 (draw), -1 (black win)
    result = board.result()
    if result == "1-0":
        value = 1
    elif result == "0-1":
        value = -1
    else:
        value = 0

    return states, value


if __name__ == "__main__":
    all_states, value = play_random_game()
    print(f"Played random game with {len(all_states)} moves, result value = {value}")
    print("Encoded tensor shape:", all_states[0].shape)
