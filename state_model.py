import torch
import torch.nn as nn


class StateMLP(nn.Module):
    """
    Input: (batch, context, 4)   # last 4 = [x, y, vx, vy]
    Output: (batch, 4)           # next state
    """

    def __init__(
        self,
        context=5,
        state_dim=4,
        hidden_dim=128,
        num_layers=3,
    ):
        super().__init__()
        self.context = context
        self.state_dim = state_dim

        input_dim = context * state_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, state_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, states):
        # states: (B, C, 4)
        b, c, d = states.shape
        x = states.view(b, c * d)  # (B, C*4)
        out = self.net(x)        # (B, 4)
        return out