import numpy as np
import logging
from typing import Optional, Dict

from .agent import PaperCompliantDDPGAgent
from .environment import PaperCompliantPlatoonEnvironment


class PlatoonTester:
    """
    Testing class for trained DDPG agent.
    """

    def __init__(self, env: PaperCompliantPlatoonEnvironment,
                 agent: PaperCompliantDDPGAgent,
                 logger: Optional[logging.Logger] = None):
        self.env = env
        self.agent = agent
        self.logger = logger or logging.getLogger(__name__)

    def test(self, num_episodes: int = 5, model_path: str = "ddpg_platoon.pth") -> Dict[str, list]:
        try:
            self.agent.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
        except FileNotFoundError:
            self.logger.error(f"Model file {model_path} not found.")
            return {}

        test_results = {
            'episode_rewards': [],
            'gap_errors': [],
            'speed_differences': [],
            'collision_count': 0
        }

        for ep in range(num_episodes):
            self.logger.info(f"Running test episode {ep + 1}/{num_episodes}")
            episode_result = self._run_test_episode()

            test_results['episode_rewards'].append(episode_result['total_reward'])
            test_results['gap_errors'].extend(episode_result['gap_errors'])
            test_results['speed_differences'].extend(episode_result['speed_differences'])
            if episode_result['collision_occurred']:
                test_results['collision_count'] += 1

        self._log_test_results(test_results)
        return test_results

    def _run_test_episode(self) -> Dict:
        state, _ = self.env.reset()
        total_reward = 0
        gap_errors = []
        speed_differences = []
        collision_occurred = False

        done = False
        step = 0

        while not done and step < self.env.max_steps:
            action = self.agent.select_action(state, add_noise=False)
            next_state, reward, done, _, reward_info = self.env.step(action)

            total_reward += reward
            gap_errors.append(abs(next_state[1]))  # gap error
            speed_differences.append(abs(next_state[2] - next_state[3]))  # speed diff

            if reward_info and reward_info.get('collision_risk', False):
                collision_occurred = True

            state = next_state
            step += 1

        return {
            'total_reward': total_reward,
            'gap_errors': gap_errors,
            'speed_differences': speed_differences,
            'collision_occurred': collision_occurred,
            'steps': step
        }

    def _log_test_results(self, results: Dict):
        if results['episode_rewards']:
            avg_reward = np.mean(results['episode_rewards'])
            avg_gap_error = np.mean(results['gap_errors'])
            avg_speed_diff = np.mean(results['speed_differences'])

            self.logger.info("=== Test Results ===")
            self.logger.info(f"Average Episode Reward: {avg_reward:.2f}")
            self.logger.info(f"Average Gap Error: {avg_gap_error:.3f} m")
            self.logger.info(f"Average Speed Difference: {avg_speed_diff:.3f} m/s")
            self.logger.info(f"Collision Count: {results['collision_count']}")
