import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional

from .networks import PaperCompliantActor, PaperCompliantCritic
from .replay_buffer import ThreeStepReplayBuffer


class PaperCompliantDDPGAgent:
    """
    DDPG Agent following the paper's specifications with modular design.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005, device: str = 'cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

        # Actor/Critic networks
        self.actor = PaperCompliantActor(state_dim, action_dim, max_action).to(device)
        self.critic = PaperCompliantCritic(state_dim, action_dim).to(device)
        self.actor_target = PaperCompliantActor(state_dim, action_dim, max_action).to(device)
        self.critic_target = PaperCompliantCritic(state_dim, action_dim).to(device)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ThreeStepReplayBuffer()

        # Exploration noise
        self.noise_std = 0.3
        self.noise_decay = 0.995
        self.min_noise = 0.05

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using actor network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)

        return action

    def train(self, batch_size: int = 256) -> Tuple[Optional[float], Optional[float]]:
        """Train the DDPG agent using 3-step TD learning."""
        if len(self.replay_buffer) < batch_size:
            return None, None

        states, actions, r_k, r_k1, r_k2, next_states, dones, _ = self.replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        r_k = r_k.to(self.device).unsqueeze(1)
        r_k1 = r_k1.to(self.device).unsqueeze(1)
        r_k2 = r_k2.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)

        # Target Q calculation
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            td_target = (
                r_k + self.gamma * r_k1 + (self.gamma ** 2) * r_k2 +
                (self.gamma ** 3) * target_q * (~dones).float()
            )

        # Critic update
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, local_model, target_model):
        """Soft update target networks."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
