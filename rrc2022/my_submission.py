"""Example policy for Real Robot Challenge 2022"""

from copy import deepcopy

import numpy as np
import torch

from rrc_2022_datasets import PolicyBase

from . import policies


class TorchBasePolicy(PolicyBase):
    def __init__(
        self,
        torch_model_path,
        action_space,
        observation_space,
        episode_length,
    ):
        self.action_space = action_space
        self.device = "cpu"

        # load torch script
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )
        self.last_cube = np.zeros(24+4+3, dtype=np.float32)
        self.last_delay = 0

    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        self.last_cube = np.zeros(24+4+3, dtype=np.float32)
        self.last_delay = 0

    def get_action(self, observation):
        if observation[self.delay_idx] != self.last_delay and (observation[self.delay_idx] < 0.5 or observation[self.delay_idx-1] > 0.6):
            self.last_cube = 0.3 * self.last_cube + 0.7 * observation[self.pose_start: self.pose_end]
            self.last_delay = observation[self.delay_idx]
        observation[self.pose_start: self.pose_end] = deepcopy(self.last_cube)
        observation = observation[self.selected_idx]
        observation = torch.tensor(observation, dtype=torch.float, device=self.device).view(1, -1)
        with torch.no_grad():
            action = self.policy(observation)
        action = action.detach().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = deepcopy(action)
        return action


class TorchPushExpertPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model = policies.get_model_path("push_expert.pt")
        pre_action_idx = list(range(3, 12))  # lift: 24~33; push: 3~12
        goal_idx = list(range(12, 15)) # lift: 33~57; push: 12~15
        pose_idx = list(range(15, 78)) + list(range(79, 97))  # lift: 57~120, 121~139, push: 15~78, 79~97
        self.selected_idx = pre_action_idx + goal_idx + pose_idx
        self.delay_idx = 16
        self.pose_start = 17
        self.pose_end = 48
        super().__init__(model, action_space, observation_space, episode_length)


class TorchLiftExpertPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model = policies.get_model_path("lift_expert.pt")
        pre_action_idx = list(range(24, 33))  # lift: 24~33; push: 3~12
        goal_idx = list(range(33, 57)) # lift: 33~57; push: 12~15
        pose_idx = list(range(57, 120)) + list(range(121, 139))  # lift: 57~120, 121~139, push: 15~78, 79~97
        self.selected_idx = pre_action_idx + goal_idx + pose_idx
        self.delay_idx = 58
        self.pose_start = 59
        self.pose_end = 90
        super().__init__(model, action_space, observation_space, episode_length)

class TorchPushMixedPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model = policies.get_model_path("push_mixed.pt")
        pre_action_idx = list(range(3, 12))  # lift: 24~33; push: 3~12
        goal_idx = list(range(12, 15)) # lift: 33~57; push: 12~15
        pose_idx = list(range(15, 78)) + list(range(79, 97))  # lift: 57~120, 121~139, push: 15~78, 79~97
        self.selected_idx = pre_action_idx + goal_idx + pose_idx
        self.delay_idx = 16
        self.pose_start = 17
        self.pose_end = 48
        super().__init__(model, action_space, observation_space, episode_length)


class TorchLiftMixedPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model = policies.get_model_path("lift_mixed.pt")
        pre_action_idx = list(range(24, 33))  # lift: 24~33; push: 3~12
        goal_idx = list(range(33, 57)) # lift: 33~57; push: 12~15
        pose_idx = list(range(57, 120)) + list(range(121, 139))  # lift: 57~120, 121~139, push: 15~78, 79~97
        self.selected_idx = pre_action_idx + goal_idx + pose_idx
        self.delay_idx = 58
        self.pose_start = 59
        self.pose_end = 90
        super().__init__(model, action_space, observation_space, episode_length)