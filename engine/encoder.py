# engine/encoder.py
import chess
import numpy as np
import torch

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    17 planes (float32):
    0-5   white P N B R Q K
    6-11  black P N B R Q K
    12    side to move (1 white, 0 black)
    13    white king-side castling
    14    white queen-side castling
    15    black king-side castling
    16    black queen-side castling
    """
    planes = np.zeros((17, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        r = 7 - (sq // 8)
        c = sq % 8
        base = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.WHITE:
            planes[base, r, c] = 1
        else:
            planes[base + 6, r, c] = 1

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
