# engine/board.py
import chess

class ChessBoard:
    def __init__(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)

    def fen(self):
        return self.board.fen()

    def legal_moves(self):
        return list(self.board.legal_moves)

    def push(self, move: chess.Move):
        self.board.push(move)

    def copy(self):
        newb = ChessBoard()
        newb.board = self.board.copy()
        return newb

    def is_game_over(self):
        return self.board.is_game_over()

    def result(self):
        return self.board.result()

    def turn(self):
        return self.board.turn
