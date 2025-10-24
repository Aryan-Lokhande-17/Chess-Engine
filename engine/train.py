import torch
import torch.nn as nn
import torch.optim as optim
from model import AlphaZeroNet
from board import encode_board  
import random

# Simulated dataset (you’ll later replace with self-play or PGN data)
def generate_fake_data(batch_size=32):
    boards = []
    policy_targets = []
    value_targets = []

    for _ in range(batch_size):
        board_tensor = encode_board()  # [17, 8, 8]
        boards.append(board_tensor)
        
        # Random move distribution (4672 possible moves)
        policy_target = torch.rand(4672)
        policy_target = policy_target / policy_target.sum()
        policy_targets.append(policy_target)

        # Random game result (between -1 loss, 0 draw, +1 win)
        value_target = torch.tensor([random.uniform(-1, 1)])
        value_targets.append(value_target)

    return (
        torch.stack(boards),
        torch.stack(policy_targets),
        torch.stack(value_targets),
    )

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroNet().to(device)

# Losses
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # For demo
    model.train()
    boards, policy_targets, value_targets = generate_fake_data(batch_size=8)
    boards, policy_targets, value_targets = (
        boards.to(device),
        policy_targets.to(device),
        value_targets.to(device),
    )

    optimizer.zero_grad()
    policy_logits, values = model(boards)

    # Cross-entropy expects class indices → use log_softmax to convert logits
    policy_loss = -torch.mean(torch.sum(policy_targets * torch.log_softmax(policy_logits, dim=1), dim=1))
    value_loss = value_loss_fn(values.squeeze(), value_targets.squeeze())

    total_loss = policy_loss + value_loss
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/5 | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")
