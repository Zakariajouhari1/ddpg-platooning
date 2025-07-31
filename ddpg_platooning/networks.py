import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperCompliantActor(nn.Module):
    """Actor network following paper specifications."""
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 200)
        self.layer4 = nn.Linear(200, 50)
        self.output_layer = nn.Linear(50, action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = torch.tanh(self.output_layer(x))
        return x * self.max_action


class PaperCompliantCritic(nn.Module):
    """Critic network following paper specifications."""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400 + action_dim, 300)
        self.layer3 = nn.Linear(300, 200)
        self.layer4 = nn.Linear(200, 50)
        self.output_layer = nn.Linear(50, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, state, action):
        x = F.relu(self.layer1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.output_layer(x)
