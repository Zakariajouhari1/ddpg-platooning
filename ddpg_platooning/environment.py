import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
from typing import Optional

from .reward_function import PaperCompliantRewardFunction

# Ensure SUMO tools are on PYTHONPATH
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


class PaperCompliantPlatoonEnvironment(gym.Env):
    """
    Platooning environment following the paper specifications.
    """

    def __init__(self, sumo_cfg_file: str, use_gui: bool = False,
                 max_steps: int = 300,
                 reward_function: Optional[PaperCompliantRewardFunction] = None):
        super().__init__()
        self.sumo_cfg = sumo_cfg_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.leader_id = "leader_0"
        self.follower_id = "follower_0"
        self.step_count = 0

        # Physical parameters from the paper
        self.d_d = 4.0
        self.L = 3.2
        self.max_acceleration = 3.5
        self.max_deceleration = -3.5
        self.delta_T = 0.25

        # Reward function
        self.reward_function = reward_function or PaperCompliantRewardFunction()

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, -50.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([200.0, 50.0, 50.0, 50.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([self.max_deceleration], dtype=np.float32),
            high=np.array([self.max_acceleration], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        self.prev_speed = None
        self.debug_mode = True

    # ---------- SUMO Helpers ----------

    def _start_sumo(self):
        """Start SUMO simulation."""
        cmd = ["sumo-gui" if self.use_gui else "sumo", "-c", self.sumo_cfg,
               "--start", "--quit-on-end", "--no-warnings", "--no-step-log"]
        traci.start(cmd)

    def _spawn_vehicles(self):
        """Spawn vehicles in simulation."""
        for _ in range(3):
            traci.simulationStep()

        existing_vehicles = traci.vehicle.getIDList()
        if self.leader_id not in existing_vehicles:
            traci.vehicle.add(self.leader_id, "highway_route", typeID="leader_vehicle",
                              departPos="120", departSpeed="15")
        if self.follower_id not in existing_vehicles:
            traci.vehicle.add(self.follower_id, "highway_route", typeID="platoon_vehicle",
                              departPos="105", departSpeed="15")
        for _ in range(5):
            traci.simulationStep()

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        vehicle_list = traci.vehicle.getIDList()
        if self.leader_id not in vehicle_list or self.follower_id not in vehicle_list:
            return np.array([10.0, 0.0, 15.0, 15.0], dtype=np.float32)

        follower_pos = traci.vehicle.getPosition(self.follower_id)[0]
        leader_pos = traci.vehicle.getPosition(self.leader_id)[0]
        follower_speed = traci.vehicle.getSpeed(self.follower_id)
        leader_speed = traci.vehicle.getSpeed(self.leader_id)

        d_actual = leader_pos - follower_pos
        gap_error = d_actual - self.L - self.d_d
        return np.array([d_actual, gap_error, follower_speed, leader_speed], dtype=np.float32)

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        try:
            traci.close()
        except:
            pass
        time.sleep(0.2)

        self._start_sumo()
        self._spawn_vehicles()

        self.step_count = 0
        self.prev_speed = None
        self.reward_function.reset_state()
        return self._get_observation(), {}

    def step(self, action):
        vehicle_list = traci.vehicle.getIDList()
        if self.follower_id not in vehicle_list:
            return self._get_observation(), -100, True, False, {}

        current_speed = traci.vehicle.getSpeed(self.follower_id)

        # Apply follower action
        new_speed = max(0.1, current_speed + action[0] * self.delta_T)
        traci.vehicle.setSpeed(self.follower_id, new_speed)

        # Leader vehicle control
        if self.leader_id in vehicle_list:
            base_speed = 15.0
            t_factor = self.step_count * 0.05
            leader_target = base_speed + 3.0 * np.sin(t_factor) * np.cos(t_factor * 0.3)
            leader_target = np.clip(leader_target, 10.0, 25.0)
            traci.vehicle.setSpeed(self.leader_id, leader_target)

        traci.simulationStep()
        self.step_count += 1

        obs = self._get_observation()

        if self.prev_speed is not None:
            reward, reward_info = self.reward_function.calculate_reward(obs, action, self.prev_speed)
        else:
            reward, reward_info = 0.0, {}

        self.prev_speed = obs[2]
        done = self.step_count >= self.max_steps
        truncated = False

        if len(traci.vehicle.getIDList()) < 2:
            done = True
            reward = -100

        return obs, reward, done, truncated, reward_info

    def close(self):
        try:
            traci.close()
        except:
            pass
