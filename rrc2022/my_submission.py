"""Example policy for Real Robot Challenge 2022"""
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

    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = observation[self.selected_idx]
        observation = torch.tensor(observation, dtype=torch.float, device=self.device).view(1, -1)
        with torch.no_grad():
            action = self.policy(observation)
        action = action.detach().numpy()[0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
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
        super().__init__(model, action_space, observation_space, episode_length)