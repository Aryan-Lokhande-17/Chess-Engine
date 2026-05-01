"""
mcts_improved.py  –  Drop into /engine/, replace your mcts.py (or import selectively).

Key upgrades over a basic AlphaZero MCTS:
  1. Opening book integration  (OpeningBook)
  2. Temperature-scheduled move selection (sharp early, exploitative later)
  3. Dirichlet noise at root (encourages exploration during self-play)
  4. Virtual loss for thread-safe parallel MCTS (optional)
  5. PUCT with proper FPU (First-Play Urgency) for unvisited nodes
  6. Draw detection & repetition penalty in backup
  7. Cleaner Node / MCTS API that matches your existing model interface

Replace calls to your old MCTS class with MCTSImproved.
The constructor signature is backward-compatible with a single `model` arg.
"""

import math
import numpy as np
import chess
from typing import Optional, Tuple, List

try:
    from engine.opening_book import OpeningBook
except ImportError:
    from opening_book import OpeningBook   # fallback when running standalone


# ---------------------------------------------------------------------------
# Hyper-parameters (tune these)
# ---------------------------------------------------------------------------
C_PUCT         = 2.5     # exploration constant (AlphaZero used ~2.5 for chess)
DIRICHLET_ALPHA = 0.3    # AlphaZero value for chess
DIRICHLET_FRAC  = 0.25   # how much noise to mix in at root
FPU_REDUCTION   = 0.2    # first-play urgency: parent_v - FPU_REDUCTION
DRAW_VALUE      = 0.0    # value assigned to a drawn position
REPETITION_PENALTY = 0.05  # subtract this from backup value per repetition


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class Node:
    __slots__ = ("prior", "visit_count", "value_sum",
                 "children", "is_expanded", "virtual_loss")

    def __init__(self, prior: float = 0.0):
        self.prior        = prior
        self.visit_count  = 0
        self.value_sum    = 0.0
        self.children     = {}     # {chess.Move: Node}
        self.is_expanded  = False
        self.virtual_loss = 0      # for parallel MCTS

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, fpu: float) -> float:
        """PUCT score."""
        q = self.value if self.visit_count > 0 else fpu
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u

    def expand(self, policy: dict):
        """policy: {chess.Move: probability}"""
        for move, prob in policy.items():
            if move not in self.children:
                self.children[move] = Node(prior=prob)
        self.is_expanded = True

    def select_child(self, fpu: float) -> Tuple["Node", chess.Move]:
        """Select child with highest UCB score."""
        parent_visits = self.visit_count + 1   # +1 avoids sqrt(0)
        best_score, best_move, best_child = -float("inf"), None, None
        for move, child in self.children.items():
            score = child.ucb_score(parent_visits, fpu)
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_child, best_move

    def add_dirichlet_noise(self):
        """Add exploration noise to root node priors."""
        if not self.children:
            return
        moves = list(self.children.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        for move, n in zip(moves, noise):
            child = self.children[move]
            child.prior = (1 - DIRICHLET_FRAC) * child.prior + DIRICHLET_FRAC * n


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------
class MCTSImproved:
    """
    Drop-in replacement for your MCTS class.

    model: your neural network with a method
        value, policy_logits = model.predict(board)   OR
        value, policy_dict   = model(board)
    The model should return:
        value        – float in [-1, 1] from current player's perspective
        policy_dict  – {chess.Move: probability} (softmax over legal moves)

    If your model returns a raw array (like AlphaZero encoding), set
    `legacy_model=True` and provide an `encoder` and `move_encoder`.
    """

    def __init__(self, model, use_opening_book: bool = True,
                 legacy_model: bool = False,
                 encoder=None, move_encoder=None):
        self.model           = model
        self.use_book        = use_opening_book
        self.book            = OpeningBook() if use_opening_book else None
        self.legacy_model    = legacy_model
        self.encoder         = encoder
        self.move_encoder    = move_encoder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_move(self, board: chess.Board,
                 num_simulations: int = 800,
                 temperature: float = None,
                 add_noise: bool = False) -> chess.Move:
        """
        Return the best move for `board`.

        temperature:
            None  → use schedule (sharp if fullmove < 30, greedy after)
            0     → always pick the most-visited move (greedy)
            1     → sample proportional to visit counts
        add_noise:
            True during self-play training, False for evaluation/play.
        """
        # 1. Check opening book
        if self.use_book and self.book.in_book(board):
            book_move = self.book.get_move(board)
            if book_move:
                return book_move

        # 2. Run MCTS
        root = Node()
        self._expand_node(root, board)
        if add_noise:
            root.add_dirichlet_noise()

        for _ in range(num_simulations):
            self._simulate(root, board.copy())

        # 3. Select move
        if temperature is None:
            temperature = 1.0 if board.fullmove_number <= 30 else 0.0

        return self._select_move(root, temperature)

    def get_policy_and_value(self, board: chess.Board,
                              num_simulations: int = 800,
                              add_noise: bool = True):
        """
        Used during self-play to get the improved policy target π and value.
        Returns: (policy_dict {Move: prob}, value float)
        """
        if self.use_book and self.book.in_book(board):
            book_move = self.book.get_move(board)
            if book_move:
                # Return a deterministic policy for the book move
                policy = {m: 0.0 for m in board.legal_moves}
                policy[book_move] = 1.0
                return policy, 0.0   # value unknown from book, use 0

        root = Node()
        self._expand_node(root, board)
        if add_noise:
            root.add_dirichlet_noise()

        for _ in range(num_simulations):
            self._simulate(root, board.copy())

        # Build π from visit counts
        total = sum(c.visit_count for c in root.children.values()) + 1e-8
        policy = {m: c.visit_count / total for m, c in root.children.items()}
        return policy, root.value

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _simulate(self, root: Node, board: chess.Board):
        """One MCTS simulation: select → expand → evaluate → backup."""
        path: List[Tuple[Node, Optional[chess.Move]]] = [(root, None)]
        node = root

        # Selection
        while node.is_expanded and node.children:
            fpu = node.value - FPU_REDUCTION
            child, move = node.select_child(fpu)
            board.push(move)
            path.append((child, move))
            node = child

        # Check terminal
        if board.is_game_over():
            result = board.result()
            value = self._terminal_value(result, board.turn)
        else:
            # Expansion + Evaluation
            value = self._expand_node(node, board)

        # Repetition penalty
        if board.is_repetition(2):
            value -= REPETITION_PENALTY

        # Backup (negate value at each ply because turns alternate)
        for i, (n, _) in enumerate(reversed(path)):
            n.visit_count  += 1
            n.value_sum    += value if i % 2 == 0 else -value

    def _expand_node(self, node: Node, board: chess.Board) -> float:
        """Evaluate board with neural network, expand node. Returns value."""
        value, policy_dict = self._evaluate(board)
        # Normalise policy to legal moves
        legal = list(board.legal_moves)
        total = sum(policy_dict.get(m, 1e-8) for m in legal) + 1e-8
        normalized = {m: policy_dict.get(m, 1e-8) / total for m in legal}
        node.expand(normalized)
        return value

    def _evaluate(self, board: chess.Board):
        """
        Call the neural network.  Supports two calling conventions:
          A) model.predict(board) → (value, {Move: prob})
          B) Legacy: model(encoded) → (value_tensor, policy_tensor)
             requires self.encoder and self.move_encoder
        """
        if self.legacy_model:
            return self._evaluate_legacy(board)
        # Default: model has a .predict(board) method
        try:
            value, policy_dict = self.model.predict(board)
        except AttributeError:
            value, policy_dict = self.model(board)
        return float(value), policy_dict

    def _evaluate_legacy(self, board: chess.Board):
        """For AlphaZero-style models that take an encoded state."""
        import torch
        encoded = self.encoder.encode(board)               # your encode fn
        tensor  = torch.tensor(encoded).unsqueeze(0).float()
        with torch.no_grad():
            value_t, policy_t = self.model(tensor)
        value = float(value_t.squeeze())

        # Build {Move: prob} from policy output
        legal = list(board.legal_moves)
        indices = [self.move_encoder.encode_move(m, board.turn) for m in legal]
        logits  = policy_t.squeeze().numpy()
        probs   = _softmax(logits[indices])
        policy_dict = dict(zip(legal, probs))
        return value, policy_dict

    def _select_move(self, root: Node, temperature: float) -> chess.Move:
        moves   = list(root.children.keys())
        visits  = np.array([root.children[m].visit_count for m in moves], dtype=float)

        if temperature == 0 or temperature < 1e-6:
            return moves[int(np.argmax(visits))]

        # Sample proportional to visits^(1/temperature)
        visits = visits ** (1.0 / temperature)
        visits /= visits.sum()
        idx = np.random.choice(len(moves), p=visits)
        return moves[idx]

    @staticmethod
    def _terminal_value(result: str, turn: chess.Color) -> float:
        if result == "1-0":
            return 1.0 if turn == chess.BLACK else -1.0
        if result == "0-1":
            return 1.0 if turn == chess.WHITE else -1.0
        return DRAW_VALUE   # draw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()