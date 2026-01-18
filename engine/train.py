# engine/train.py
import torch
import torch.nn as nn
import torch.optim as optim

from model import AlphaZeroNet
from self_play import self_play_game
from move_encoding import POLICY_SIZE

def train_one_iteration(model, device="cpu", games=5, sims=100, epochs=2, batch_size=32):
    # 1) Generate data from self-play
    replay = []
    for g in range(games):
        data, res = self_play_game(model, device=device, sims=sims)
        replay.extend(data)
        print(f"Self-play game {g+1}/{games} done -> {res}, samples: {len(data)}")

    # 2) Train on replay buffer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    value_loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0

        # shuffle
        perm = torch.randperm(len(replay))

        for i in range(0, len(replay), batch_size):
            idx = perm[i:i+batch_size]
            batch = [replay[j] for j in idx]

            states = torch.stack([b[0] for b in batch]).to(device)         # (B,17,8,8)
            target_pi = torch.stack([b[1] for b in batch]).to(device)      # (B,4096)
            target_v = torch.stack([b[2] for b in batch]).to(device)       # (B,1)

            optimizer.zero_grad()
            policy_logits, pred_v = model(states)

            # ✅ policy loss (distribution cross entropy)
            policy_loss = -torch.mean(torch.sum(target_pi * torch.log_softmax(policy_logits, dim=1), dim=1))

            # ✅ value loss
            value_loss = value_loss_fn(pred_v, target_v)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        print(f"Epoch {ep+1}/{epochs} | PolicyLoss: {total_policy_loss:.4f} | ValueLoss: {total_value_loss:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaZeroNet(in_channels=17, num_res_blocks=4, channels=128, policy_size=POLICY_SIZE).to(device)

    print("Training on device:", device)
    train_one_iteration(model, device=device, games=3, sims=80, epochs=2)

    # Save checkpoint
    torch.save(model.state_dict(), "data/checkpoints/latest.pt")
    print("✅ Saved model to data/checkpoints/latest.pt")

if __name__ == "__main__":
    main()
