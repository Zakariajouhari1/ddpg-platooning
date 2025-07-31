import random
from collections import deque
import numpy as np
import torch

class ThreeStepReplayBuffer:
    """3-step TD learning replay buffer as described in the paper."""
    def __init__(self, capacity: int = 1_000_000, gamma: float = 0.99):
        self.capacity = capacity
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.temp_buffer = deque(maxlen=3)

    def push(self, state, action, reward, next_state, done, v_ego_prev):
        """
        Store a new experience in the buffer with 3-step returns.
        """
        self.temp_buffer.append((state, action, reward, next_state, done, v_ego_prev))

        if len(self.temp_buffer) >= 3:
            s_k, a_k, r_k, _, _, v_prev_k = self.temp_buffer[0]
            _, _, r_k1, _, _, _ = self.temp_buffer[1]
            _, _, r_k2, s_k3, done_k3, _ = self.temp_buffer[2]
            self.buffer.append((s_k, a_k, r_k, r_k1, r_k2, s_k3, done_k3, v_prev_k))

        # Handle episode end flush
        if done and len(self.temp_buffer) > 1:
            for i in range(1, len(self.temp_buffer)):
                if i == 1:
                    s_k, a_k, r_k, _, _, v_prev_k = self.temp_buffer[i - 1]
                    _, _, r_k1, s_k2, done_k2, _ = self.temp_buffer[i]
                    self.buffer.append((s_k, a_k, r_k, r_k1, 0.0, s_k2, done_k2, v_prev_k))
                else:
                    s_k, a_k, r_k, s_k1, done_k1, v_prev_k = self.temp_buffer[i - 1]
                    self.buffer.append((s_k, a_k, r_k, 0.0, 0.0, s_k1, done_k1, v_prev_k))
            self.temp_buffer.clear()

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences, converted efficiently to tensors.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        batch = random.sample(self.buffer, batch_size)

        # Vectorized NumPy → Torch conversion (avoids PyTorch warning)
        states = torch.from_numpy(np.array([e[0] for e in batch], dtype=np.float32))
        actions = torch.from_numpy(np.array([e[1] for e in batch], dtype=np.float32))
        r_k = torch.from_numpy(np.array([e[2] for e in batch], dtype=np.float32))
        r_k1 = torch.from_numpy(np.array([e[3] for e in batch], dtype=np.float32))
        r_k2 = torch.from_numpy(np.array([e[4] for e in batch], dtype=np.float32))
        next_states = torch.from_numpy(np.array([e[5] for e in batch], dtype=np.float32))
        dones = torch.from_numpy(np.array([e[6] for e in batch], dtype=np.bool_))
        v_ego_prevs = torch.from_numpy(np.array([e[7] for e in batch], dtype=np.float32))

        return states, actions, r_k, r_k1, r_k2, next_states, dones, v_ego_prevs

    def __len__(self):
        return len(self.buffer)
