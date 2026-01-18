# engine/move_encoding.py
import chess

BASE_POLICY_SIZE = 64 * 64  # 4096
PROMO_SIZE = 64 * 4         # 256
POLICY_SIZE = BASE_POLICY_SIZE + PROMO_SIZE  # 4352

PROMO_TO_INDEX = {
    chess.QUEEN: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
}

INDEX_TO_PROMO = {v: k for k, v in PROMO_TO_INDEX.items()}


def move_to_index(move: chess.Move) -> int:
    """
    Normal move: from*64 + to (0..4095)
    Promotion: 4096 + from*4 + promo_type (4096..4351)
    """
    if move.promotion is not None:
        promo_idx = PROMO_TO_INDEX[move.promotion]
        return BASE_POLICY_SIZE + move.from_square * 4 + promo_idx

    return move.from_square * 64 + move.to_square
