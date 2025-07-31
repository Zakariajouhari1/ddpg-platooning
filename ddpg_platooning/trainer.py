import numpy as np
from typing import Optional, Dict, Tuple

from .agent import PaperCompliantDDPGAgent
from .environment import PaperCompliantPlatoonEnvironment
from .utils.logger import setup_logger
from .utils.plotter import plot_training_results


class PlatoonTrainer:
    """
    Modular training class for DDPG platooning.
    """

    def __init__(self, env: PaperCompliantPlatoonEnvironment,
                 agent: PaperCompliantDDPGAgent,
                 logger=None):
        self.env = env
        self.agent = agent
        self.logger = logger or setup_logger(__name__, log_file="training.log")

        # Metrics
        self.episode_rewards = []
        self.critic_losses = []
        self.actor_losses = []
        self.reward_components = []

    def train(self, episodes: int = 2000, batch_size: int = 256,
              save_interval: int = 100, model_path: str = "ddpg_platoon.pth") -> Dict[str, list]:
        self.logger.info(f"Starting training for {episodes} episodes...")

        for episode in range(episodes):
            try:
                ep_reward, ep_info = self._run_episode(batch_size)
                self.episode_rewards.append(ep_reward)
                self.reward_components.append(ep_info)

                if episode % 10 == 0:
                    self._log_progress(episode)

                if episode % save_interval == 0 and episode > 0:
                    self.agent.save(f"{model_path}_{episode}")
                    self.logger.info(f"Saved model checkpoint at episode {episode}")

            except Exception as e:
                self.logger.error(f"Error in episode {episode}: {e}")
                continue

        self.agent.save(model_path)
        self.logger.info(f"Final model saved to {model_path}")

        return {
            'episode_rewards': self.episode_rewards,
            'critic_losses': self.critic_losses,
            'actor_losses': self.actor_losses,
            'reward_components': self.reward_components
        }

    def _run_episode(self, batch_size: int) -> Tuple[float, Dict]:
        state, _ = self.env.reset()
        ep_reward = 0
        ep_info = {'main_rewards': [], 'jerk_penalties': [], 'RTG_values': [], 'collision_risks': 0}
        done = False
        step_in_ep = 0

        while not done and step_in_ep < self.env.max_steps:
            action = self.agent.select_action(state, add_noise=True)
            next_state, reward, done, _, reward_info = self.env.step(action)

            v_ego_prev = state[2] if hasattr(self.env, 'prev_speed') and self.env.prev_speed is not None else state[2]
            self.agent.replay_buffer.push(state, action, reward, next_state, done, v_ego_prev)

            if len(self.agent.replay_buffer) > batch_size * 2:
                critic_loss, actor_loss = self.agent.train(batch_size)
                if critic_loss is not None and actor_loss is not None:
                    self.critic_losses.append(critic_loss)
                    self.actor_losses.append(actor_loss)

            ep_reward += reward
            if reward_info:
                ep_info['main_rewards'].append(reward_info.get('main_reward', 0))
                ep_info['jerk_penalties'].append(reward_info.get('jerk_penalty', 0))
                ep_info['RTG_values'].append(reward_info.get('RTG', 0))
                if reward_info.get('collision_risk', False):
                    ep_info['collision_risks'] += 1

            state = next_state
            step_in_ep += 1

        return ep_reward, ep_info

    def _log_progress(self, episode: int):
        if len(self.episode_rewards) >= 10:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_critic_loss = np.mean(self.critic_losses[-100:]) if self.critic_losses else 0
            avg_actor_loss = np.mean(self.actor_losses[-100:]) if self.actor_losses else 0

            self.logger.info(
                f"Episode {episode:4d} | Avg Reward: {avg_reward:8.2f} | "
                f"Critic Loss: {avg_critic_loss:8.4f} | Actor Loss: {avg_actor_loss:8.4f} | "
                f"Buffer Size: {len(self.agent.replay_buffer):6d} | Noise: {self.agent.noise_std:.3f}"
            )

    def plot_training_results(self, save_path: str = "training_results.png"):
        plot_training_results(
            episode_rewards=self.episode_rewards,
            critic_losses=self.critic_losses,
            actor_losses=self.actor_losses,
            reward_components=self.reward_components,
            save_path=save_path
        )
