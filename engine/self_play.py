# engine/self_play.py
import chess
import torch
import numpy as np

from engine.encoder import board_to_tensor
from engine.mcts import MCTS
from engine.move_encoding import POLICY_SIZE, move_to_index


def result_to_value(board: chess.Board) -> float:
    res = board.result()
    if res == "1-0":
        return 1.0
    elif res == "0-1":
        return -1.0
    return 0.0


def pick_move_from_pi(board: chess.Board, pi: np.ndarray, temperature: float = 1.0):
    legal_moves = list(board.legal_moves)

    probs = np.array([pi[move_to_index(m)] for m in legal_moves], dtype=np.float32)

    if probs.sum() <= 1e-9:
        probs = np.ones(len(legal_moves), dtype=np.float32) / len(legal_moves)
    else:
        probs = probs / probs.sum()

    # temperature=0 => pick max
    if temperature <= 1e-6:
        return legal_moves[int(np.argmax(probs))]

    # temperature sampling
    probs = np.power(probs, 1.0 / temperature)
    probs = probs / probs.sum()

    idx = np.random.choice(len(legal_moves), p=probs)
    return legal_moves[idx]


def self_play_game(
    model,
    device="cpu",
    sims=200,
    max_moves=220,
    temperature_moves=12,
    resign_threshold=-0.90,
    resign_patience=6,
):
    board = chess.Board()
    mcts = MCTS(model, device=device, simulations=sims)

    samples = []  # (state_tensor, pi_np, turn)

    resign_count = 0

    while not board.is_game_over() and board.fullmove_number < max_moves:
        state = board_to_tensor(board)

        best_move, pi = mcts.run(board)

        # store training sample
        samples.append((state, pi, board.turn))

        # quick value estimate for resign logic
        model.eval()
        with torch.no_grad():
            x = state.unsqueeze(0).to(device)
            _, v = model(x)
            v = float(v.item())

        # resign streak check (from side-to-move perspective)
        if v < resign_threshold:
            resign_count += 1
        else:
            resign_count = 0

        if resign_count >= resign_patience:
            # side to move resigns => opponent wins
            if board.turn == chess.WHITE:
                return finalize_samples(samples, z=-1.0), "0-1"
            else:
                return finalize_samples(samples, z=1.0), "1-0"

        # temperature schedule
        if board.fullmove_number <= temperature_moves:
            move = pick_move_from_pi(board, pi, temperature=1.0)
        else:
            move = best_move

        board.push(move)

    # final result if game ended or max_moves reached
    z = result_to_value(board)
    return finalize_samples(samples, z=z), board.result()


def finalize_samples(samples, z: float):
    dataset = []
    for state, pi, turn in samples:
        value = z if turn == chess.WHITE else -z
        dataset.append((
            state,
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32)
        ))
    return dataset
