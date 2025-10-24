import torch
import chess
import numpy as np

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a python-chess Board into a (17, 8, 8) tensor of 0/1 floats.

    Planes:
    0–5: White pieces
    6–11: Black pieces
    12: Side to move (1 if white, else 0)
    13–16: Castling rights (WK, WQ, BK, BQ)
    """

    planes = np.zeros((17, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        base = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.WHITE:
            planes[base, row, col] = 1
        else:
            planes[base + 6, row, col] = 1

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    return torch.tensor(planes, dtype=torch.float32)

if __name__ == "__main__":
    board = chess.Board()
    tensor = board_to_tensor(board)
    print("Tensor shape:", tensor.shape)
    print("White to move plane mean:", tensor[12].mean())
