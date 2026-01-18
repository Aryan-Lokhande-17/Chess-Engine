from fastapi import FastAPI
from pydantic import BaseModel
import chess
import torch

from fastapi.middleware.cors import CORSMiddleware

from engine.model import AlphaZeroNet
from engine.mcts import MCTS
from engine.move_encoding import POLICY_SIZE

app = FastAPI()

# ✅ CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ BIG MODEL must match training
model = AlphaZeroNet(
    in_channels=17,
    num_res_blocks=8,
    channels=256,
    policy_size=POLICY_SIZE
).to(device)

model.load_state_dict(torch.load("data/checkpoints/latest.pt", map_location=device))
model.eval()

mcts = MCTS(model=model, device=device, simulations=400, c_puct=1.5)

class MoveRequest(BaseModel):
    fen: str


@app.get("/")
def home():
    return {"status": "Chess Engine API running ✅", "device": device}


@app.post("/bestmove")
def bestmove(req: MoveRequest):
    board = chess.Board(req.fen)

    if board.is_game_over():
        return {"bestmove": None, "result": board.result()}

    move, _ = mcts.run(board)
    return {"bestmove": move.uci()}
