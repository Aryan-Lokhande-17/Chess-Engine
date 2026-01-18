console.log("✅ script.js loaded");

const game = new Chess();

const statusEl = document.getElementById("status");
const fenBox = document.getElementById("fenBox");
const resetBtn = document.getElementById("resetBtn");
const sideSelect = document.getElementById("sideSelect");

let humanSide = "white";
let aiThinking = false; // ✅ lock to stop loops

let board = Chessboard("board", {
  draggable: true,
  position: "start",
  pieceTheme: "pieces/{piece}.png",
  onDrop: onDrop,
});

function isHumanTurn() {
  if (humanSide === "white") return game.turn() === "w";
  return game.turn() === "b";
}

function updateUI() {
  fenBox.value = game.fen();

  if (game.game_over()) {
    statusEl.innerText = "Game finished ✅";
    return;
  }

  if (aiThinking) {
    statusEl.innerText = "AI thinking... 🤖";
  } else {
    statusEl.innerText = isHumanTurn() ? "Your turn" : "AI turn";
  }
}

async function requestAIMove() {
  // ✅ stop if already thinking
  if (aiThinking) return;

  // ✅ stop if game over or it's not AI's turn
  if (game.game_over()) return;
  if (isHumanTurn()) return;

  aiThinking = true;
  updateUI();

  try {
    const res = await fetch("http://127.0.0.1:8000/bestmove", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen: game.fen() }),
    });

    const data = await res.json();

    if (!data.bestmove) {
      aiThinking = false;
      updateUI();
      return;
    }

    const uci = data.bestmove;

    game.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci.length === 5 ? uci[4] : undefined,
    });

    board.position(game.fen());
  } catch (err) {
    console.error("❌ AI move fetch failed:", err);
  }

  aiThinking = false;
  updateUI();
}

function onDrop(source, target) {
  // ✅ prevent moves while AI thinking
  if (aiThinking) return "snapback";

  // ✅ prevent human moving on AI turn
  if (!isHumanTurn()) return "snapback";

  const move = game.move({
    from: source,
    to: target,
    promotion: "q",
  });

  if (move === null) return "snapback";

  board.position(game.fen());
  updateUI();

  // ✅ AI responds ONCE
  setTimeout(requestAIMove, 200);
}

resetBtn.addEventListener("click", () => {
  game.reset();

  humanSide = sideSelect.value;
  board.orientation(humanSide);
  board.start();

  aiThinking = false;
  updateUI();

  // ✅ if human plays black, AI plays first ONCE
  setTimeout(requestAIMove, 200);
});

sideSelect.addEventListener("change", () => {
  humanSide = sideSelect.value;
  board.orientation(humanSide);
  updateUI();
});

// initial setup
humanSide = sideSelect.value;
board.orientation(humanSide);
updateUI();
