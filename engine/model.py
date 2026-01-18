import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard 2-conv residual block:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> (add x) -> ReLU
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual        # skip/identity connection
        out = F.relu(out, inplace=True)
        return out

class AlphaZeroNet(nn.Module):
    """
    AlphaZero-like network.

    Args:
        in_channels: number of input planes (e.g., 17 or 18)
        num_res_blocks: how many residual blocks in the trunk (5-20 typical)
        channels: number of filters in the trunk conv layers (e.g., 128 or 256)
        policy_size: output size of policy vector (e.g., 4672)
    """
    def __init__(self, in_channels=17, num_res_blocks=6, channels=128, policy_size=4096):
        super().__init__()

        # Initial conv layer to expand channels
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        # Residual trunk
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head:
        # 1x1 conv reduces channels -> small flatten -> FC to policy size (logits)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_size)

        # Value head:
        # 1x1 conv to 1 channel -> FC -> FC -> tanh scalar
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialize weights (good practice)
        self._init_weights()

    def _init_weights(self):
        # Kaiming / He initialization for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: tensor of shape (batch, in_channels, 8, 8)
        returns:
            policy_logits: (batch, policy_size)
            value: (batch, 1) where each value is between -1 and 1 (tanh)
        """
        # trunk
        out = self.conv_in(x)              # (B, channels, 8, 8)
        out = self.bn_in(out)
        out = F.relu(out, inplace=True)

        out = self.res_blocks(out)         # (B, channels, 8, 8)

        # policy head
        p = self.policy_conv(out)          # (B, 2, 8, 8)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(p.size(0), -1)          # flatten -> (B, 2*8*8)
        policy_logits = self.policy_fc(p)  # (B, policy_size)

        # value head
        v = self.value_conv(out)           # (B, 1, 8, 8)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)          # (B, 8*8)
        v = F.relu(self.value_fc1(v), inplace=True)  # (B, 256)
        value = torch.tanh(self.value_fc2(v))        # (B, 1) in [-1,1]

        return policy_logits, value

# Quick local test when running this file directly
if __name__ == "__main__":
    # Basic sanity check: shapes and forward pass
    net = AlphaZeroNet(in_channels=17, num_res_blocks=4, channels=128, policy_size=4672)
    print("Model created:", net)

    # random input: batch_size=2, 17 planes, 8x8
    x = torch.randn(2, 17, 8, 8)
    policy_logits, value = net(x)

    print("policy_logits shape:", policy_logits.shape)  # expected (2, 4672)
    print("value shape:", value.shape)                  # expected (2, 1)
    # optionally ensure no NaNs and reasonable ranges
    print("policy_logits finite:", torch.isfinite(policy_logits).all().item())
    print("value range:", float(value.min().item()), float(value.max().item()))
