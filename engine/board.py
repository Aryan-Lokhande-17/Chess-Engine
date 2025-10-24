import torch
import chess

def encode_board(board=None):
    """
    Encode a chess.Board() into a (17, 8, 8) tensor representation.
    If no board is given, start from initial position.
    """
    if board is None:
        board = chess.Board()

    planes = torch.zeros((17, 8, 8), dtype=torch.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Encode piece positions (6 planes for white, 6 for black)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        plane_index = piece_map[piece.piece_type]
        if piece.color == chess.WHITE:
            planes[plane_index, row, col] = 1
        else:
            planes[6 + plane_index, row, col] = 1

    # Side to move
    planes[12, :, :] = int(board.turn)

    # Castling rights
    planes[13, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
    planes[14, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
    planes[15, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
    planes[16, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))

    return planes
