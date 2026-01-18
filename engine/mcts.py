# engine/mcts.py
import math
import numpy as np
import torch
import chess

from engine.encoder import board_to_tensor
from engine.move_encoding import POLICY_SIZE, move_to_index


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


class Node:
    def __init__(self, board: chess.Board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = float(prior)

        self.children = {}  # move -> Node
        self.N = 0          # visits
        self.W = 0.0        # total value
        self.Q = 0.0        # mean value

    def is_expanded(self):
        return len(self.children) > 0


class MCTS:
    def __init__(self, model, device="cpu", simulations=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def run(self, board: chess.Board):
        root = Node(board.copy())

        for _ in range(self.simulations):
            node = root
            path = [node]

            # Selection
            while node.is_expanded() and not node.board.is_game_over():
                move, node = self.select_child(node)
                path.append(node)

            # Expansion + Evaluation
            value = self.expand_and_evaluate(node)

            # Backprop
            self.backpropagate(path, value)

        # Choose best move by visits
        move_visits = [(m, child.N) for m, child in root.children.items()]
        move_visits.sort(key=lambda x: x[1], reverse=True)

        best_move = move_visits[0][0]

        # Build π (visit distribution) over POLICY_SIZE
        pi = np.zeros(POLICY_SIZE, dtype=np.float32)
        total = sum(v for _, v in move_visits) + 1e-9
        for move, visits in move_visits:
            pi[move_to_index(move)] = visits / total

        return best_move, pi

    def select_child(self, node: Node):
        best_score = -1e9
        best_move = None
        best_child = None

        sqrt_parent = math.sqrt(node.N + 1)

        for move, child in node.children.items():
            u = self.c_puct * child.prior * (sqrt_parent / (1 + child.N))
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand_and_evaluate(self, node: Node):
        # Terminal -> true result
        if node.board.is_game_over():
            res = node.board.result()
            if res == "1-0":
                return 1.0
            elif res == "0-1":
                return -1.0
            else:
                return 0.0

        x = board_to_tensor(node.board).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(x)
            policy_logits = policy_logits.squeeze(0).cpu().numpy()  # (POLICY_SIZE,)
            value = float(value.item())

        legal_moves = list(node.board.legal_moves)

        moves = []
        priors_raw = []

        for m in legal_moves:
            idx = move_to_index(m)
            moves.append(m)
            priors_raw.append(policy_logits[idx])

        priors = softmax(np.array(priors_raw, dtype=np.float32))

        # Dirichlet noise at root for exploration
        if node.parent is None:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise

        # Expand
        for m, p in zip(moves, priors):
            b2 = node.board.copy()
            b2.push(m)
            node.children[m] = Node(b2, parent=node, prior=float(p))

        return value

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value
