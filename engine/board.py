import chess

class ChessBoard:
    def __init__(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)

    def get_fen(self):
        #gives current fen
        return self.board.fen()

    def legal_moves(self):
        #returns all legal moves
        return [move.uci() for move in self.board.legal_moves]

    def push_move(self, move_uci):
        #Make the legal move on board
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            raise ValueError(f"Illegal move: {move_uci}")

    def is_game_over(self):
        #checks if checkmate or not
        return self.board.is_game_over()

    def result(self):
        #returns result like0-1, 1-0 , 1/2 - 1/2, etc.
        return self.board.result()
