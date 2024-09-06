from typing import Any, Tuple

import torch

# from skrl.envs.wrappers.torch.base import Wrapper
from multimodal_gym.utils.skrl.base_wrapper import Wrapper

from multimodal_gym.utils.frame_stack import LazyFrames


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any, obs_type) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None
        self._obs_type = obs_type
        # self._observation_space = self._observation_space
        self._observation_space = self._observation_space


    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, reward, terminated, truncated, info = self._env.step(actions)

        # convert LazyFrames to tensor
        self._obs_dict = self.convert_to_tensor(self._obs_dict)

        return self._obs_dict, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            self._obs_dict, info = self._env.reset()
            self._reset_once = False
        else:
            info = None

        self._obs_dict = self.convert_to_tensor(self._obs_dict)

        return self._obs_dict, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()

    def convert_to_tensor(self, obs):
        # convert LazyFrames to tensor
        if isinstance(obs, LazyFrames):
            # Concatenate along the last dimension from list of torch.Size([128, 80, 80, 3]) to ([128, 80, 80, 12])
            frames = obs._frames           
            obs = torch.cat(frames, dim=-1)
            if self._obs_type == "concat":
                obs = obs[:,8:]
        return obs