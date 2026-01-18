# engine/train_iterative.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

from engine.model import AlphaZeroNet
from engine.self_play import self_play_game
from engine.replay_buffer import ReplayBuffer
from engine.move_encoding import POLICY_SIZE


def train_on_buffer(model, buffer, device="cpu", epochs=6, batch_size=96, lr=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        policy_loss_sum = 0.0
        value_loss_sum = 0.0

        steps = max(1, len(buffer) // batch_size)

        for _ in range(steps):
            states, target_pi, target_v = buffer.sample_batch(batch_size)

            states = states.to(device)
            target_pi = target_pi.to(device)
            target_v = target_v.to(device)

            optimizer.zero_grad()
            policy_logits, pred_v = model(states)

            # Policy loss (target π distribution)
            policy_loss = -torch.mean(
                torch.sum(target_pi * torch.log_softmax(policy_logits, dim=1), dim=1)
            )

            # Value loss
            value_loss = value_loss_fn(pred_v, target_v)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()

        print(f"  Epoch {ep+1}/{epochs} | PolicyLoss={policy_loss_sum:.3f} | ValueLoss={value_loss_sum:.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("data/checkpoints", exist_ok=True)
    os.makedirs("data/selfplay", exist_ok=True)

    # ✅ BIG MODEL UPGRADE
    model = AlphaZeroNet(
        in_channels=17,
        num_res_blocks=8,
        channels=256,
        policy_size=POLICY_SIZE
    ).to(device)

    ckpt_path = "data/checkpoints/latest.pt"
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print("✅ Loaded checkpoint:", ckpt_path)
        except Exception:
            print("⚠️ Checkpoint mismatch. Delete latest.pt and replay.pt then re-run.")
            return

    buffer = ReplayBuffer(max_size=100000)  # ✅ bigger memory for stronger net
    buffer_path = "data/selfplay/replay.pt"
    buffer.load(buffer_path)
    print("✅ Replay buffer loaded. Samples:", len(buffer))

    # ✅ Iteration settings (stronger)
    iterations = 10
    games_per_iter = 12
    sims = 500

    # ✅ Self-play behaviour (stronger + decisive)
    max_moves = 180
    temperature_moves = 8
    resign_threshold = -0.85
    resign_patience = 6

    # ✅ Training behaviour
    epochs = 6
    batch_size = 96
    lr = 5e-4

    print(f"\n🔥 Training on device: {device}")
    print(f"Net: blocks=8 | channels=256 | policy={POLICY_SIZE}\n")

    for it in range(1, iterations + 1):
        print(f"\n================ ITERATION {it}/{iterations} ================")

        # 1) Self-play
        new_samples = 0
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}

        for g in range(games_per_iter):
            data, res = self_play_game(
                model,
                device=device,
                sims=sims,
                max_moves=max_moves,
                temperature_moves=temperature_moves,
                resign_threshold=resign_threshold,
                resign_patience=resign_patience,
            )

            results[res] = results.get(res, 0) + 1
            buffer.add_samples(data)
            new_samples += len(data)

            print(f"  Self-play {g+1}/{games_per_iter} -> {res} | added {len(data)} samples")

        print(f"✅ Iter {it} results summary: {results}")
        print(f"✅ Added samples: {new_samples}")
        print("✅ Buffer size:", len(buffer))

        # 2) Train
        train_on_buffer(model, buffer, device=device, epochs=epochs, batch_size=batch_size, lr=lr)

        # 3) Save
        torch.save(model.state_dict(), ckpt_path)
        buffer.save(buffer_path)

        print("Saved checkpoint:", ckpt_path)
        print("Saved replay buffer:", buffer_path)

    print("\nDone. Play in UI now!")


if __name__ == "__main__":
    main()
