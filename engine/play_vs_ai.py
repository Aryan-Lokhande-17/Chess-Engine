# engine/play_vs_ai.py
import chess
import torch
import numpy as np

from engine.model import AlphaZeroNet
from engine.mcts import MCTS
from engine.move_encoding import POLICY_SIZE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AlphaZeroNet(
        in_channels=17,
        num_res_blocks=4,
        channels=128,
        policy_size=POLICY_SIZE
    ).to(device)

    model.load_state_dict(torch.load("data/checkpoints/latest.pt", map_location=device))
    model.eval()

    board = chess.Board()

    mcts = MCTS(
        model=model,
        device=device,
        simulations=200,
        c_puct=1.5
    )

    print("✅ You are WHITE.")
    print("Moves: e2e4, g1f3, e1g1 (castle), e7e8q (promotion)")
    print(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move_str = input("\nYour move: ").strip()
            try:
                move = chess.Move.from_uci(move_str)
            except Exception:
                print("❌ Invalid move format.")
                continue

            if move not in board.legal_moves:
                print("❌ Illegal move.")
                continue

            board.push(move)
            print(board)

        else:
            print("\n🤖 Engine thinking...")
            best_move, pi = mcts.run(board)

            # opening exploration: first 8 full moves
            if board.fullmove_number <= 8:
                legal_moves = list(board.legal_moves)
                probs = np.array([pi[move_to_index(m)] for m in legal_moves], dtype=np.float32)

                if probs.sum() <= 1e-9:
                    probs = np.ones(len(legal_moves), dtype=np.float32) / len(legal_moves)
                else:
                    probs = probs / probs.sum()

                chosen = np.random.choice(len(legal_moves), p=probs)
                move = legal_moves[chosen]
            else:
                move = best_move

            board.push(move)
            print(f"🤖 Engine played: {move.uci()}")
            print(board)

    print("\n✅ Game over:", board.result())


if __name__ == "__main__":
    main()
