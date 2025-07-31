import numpy as np
from typing import Tuple, Dict

class PaperCompliantRewardFunction:
    """
    Implements the exact reward function from the paper (Section 4.3),
    with state space normalization.
    """

    def __init__(self, L=3.2, d_d=4.0, delta_T=0.25,
                 RTG_min=2.0, RTG_max=4.0, alpha=0.1,
                 a_max_acc=3.5, a_max_dec=-3.5, epsilon=1e-5,
                 d_max=100.0, v_max=33.33):
        self.L = L
        self.d_d = d_d
        self.delta_T = delta_T
        self.RTG_min = RTG_min
        self.RTG_max = RTG_max
        self.alpha = alpha
        self.a_max_acc = a_max_acc
        self.a_max_dec = a_max_dec
        self.epsilon = epsilon
        self.d_max = d_max
        self.v_max = v_max
        self.delta_a_max = a_max_acc - a_max_dec
        self.delta_RTG = 0.5 * (RTG_max - RTG_min)
        self.reset_state()

    def reset_state(self):
        self.prev_gap_error = None
        self.prev_acceleration = 0.0

    def normalize_state(self, d_actual, e_curr, v_ego, v_leader):
        d_norm = (d_actual - self.d_d) / self.d_max
        e_norm = e_curr / self.d_max
        v_ego_norm = v_ego / self.v_max
        v_leader_norm = v_leader / self.v_max
        return [d_norm, e_norm, v_ego_norm, v_leader_norm]

    def calculate_effective_distance(self, d_actual, v_ego_current, v_ego_prev):
        delta_v_ego = v_ego_current - v_ego_prev
        return max(0.0, d_actual - delta_v_ego * self.delta_T)

    def calculate_gap_error(self, d):
        return d - self.L - self.d_d

    def calculate_GED(self, current_gap_error):
        if self.prev_gap_error is None:
            return 0.0
        return abs(current_gap_error) - abs(self.prev_gap_error)

    def calculate_PED(self, GED):
        if self.prev_gap_error is None:
            return 0.0
        denom = abs(self.prev_gap_error) + self.epsilon
        return abs(GED / denom)

    def calculate_RTG(self, gap_error, speed_diff):
        return max(abs(gap_error), self.epsilon) / max(abs(speed_diff), self.epsilon)

    def calculate_RTG_penalty(self, RTG):
        return (RTG - self.delta_RTG) / self.delta_RTG

    def calculate_main_reward(self, gap_error, speed_diff, v_ego_current, v_ego_prev, d_actual):
        d_effective = self.calculate_effective_distance(d_actual, v_ego_current, v_ego_prev)
        effective_gap_error = self.calculate_gap_error(d_effective)
        GED = self.calculate_GED(effective_gap_error)
        PED = self.calculate_PED(GED)
        RTG = self.calculate_RTG(effective_gap_error, speed_diff)

        if GED > 0:
            reward = -1.0
        else:
            if self.RTG_min <= RTG <= self.RTG_max:
                reward = PED
            else:
                reward = -self.calculate_RTG_penalty(RTG)

        return reward, effective_gap_error, RTG

    def calculate_jerk_penalty(self, current_acceleration):
        jerk = (current_acceleration - self.prev_acceleration) / self.delta_T
        return -self.alpha * self.delta_T * abs(jerk) / self.delta_a_max

    def calculate_reward(self, state, action, v_ego_prev) -> Tuple[float, Dict[str, float]]:
        d_actual, e_curr , v_ego_current, v_leader = state
        current_acceleration = action[0]
        speed_diff = v_leader - v_ego_current

        # Normalize state here if needed externally
        normalized_state = self.normalize_state(d_actual, e_curr, v_ego_current, v_leader)

        main_reward, effective_gap_error, RTG = self.calculate_main_reward(
            self.calculate_gap_error(d_actual), speed_diff, v_ego_current, v_ego_prev, d_actual
        )
        jerk_penalty = self.calculate_jerk_penalty(current_acceleration)
        total_reward = main_reward + jerk_penalty

        self.prev_gap_error = effective_gap_error
        self.prev_acceleration = current_acceleration

        collision_distance = d_actual - self.L
        if collision_distance < 1.0:
            total_reward = -10.0

        return total_reward, {
            'main_reward': main_reward,
            'jerk_penalty': jerk_penalty,
            'total_reward': total_reward,
            'effective_gap_error': effective_gap_error,
            'RTG': RTG,
            'collision_risk': collision_distance < 1.0,
            'normalized_state': normalized_state
        }
