from board import ChessBoard

def test_board():
    board = ChessBoard()
    print("Starting FEN:", board.get_fen())

    moves = board.legal_moves()
    print("Legal moves from start:", moves)

    board.push_move(moves[0])
    print("New FEN:", board.get_fen())

    print("Game over?", board.is_game_over())

if __name__ == "__main__":
    test_board()
