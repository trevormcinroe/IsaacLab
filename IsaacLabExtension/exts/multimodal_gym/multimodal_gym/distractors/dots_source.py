import numpy as np
from typing import NamedTuple

import cv2

from multimodal_gym.distractors.abstract_source import AbstractDistractionSource
from multimodal_gym.distractors.dots_behaviour.constant_dots import ConstantDots
from multimodal_gym.distractors.dots_behaviour.episode_dots import EpisodeDotsSource
from multimodal_gym.distractors.dots_behaviour.linear_dots import LinearDotsSource
from multimodal_gym.distractors.dots_behaviour.random_dots import RandomDotsSource


class Limits(NamedTuple):
    low: float
    high: float


class DotsSource(AbstractDistractionSource):
    BEHAVIOURS = {
        "constant": ConstantDots,
        "episode": EpisodeDotsSource,
        "random": RandomDotsSource,
        "linear": LinearDotsSource,
    }

    def __init__(self, seed: int, shape: tuple[int, int], dots_behaviour: str, num_dots=12, dots_size=0.12):
        super().__init__(seed=seed)
        self.shape = shape
        self.num_dots = num_dots
        self.dots_size = dots_size
        self.x_lim = Limits(0.05, 0.95)
        self.y_lim = Limits(0.05, 0.95)

        self.dots_behaviour = self.BEHAVIOURS[dots_behaviour]()
        self.dots_state = self.dots_behaviour.init_state(self.num_dots, self.x_lim, self.y_lim, self._rng)
        self.positions = self.dots_behaviour.get_positions(self.dots_state)
        self.dots_parameters = self.init_dots()

    def get_info(self):
        info = super().get_info()
        return {
            **info,
            "num_dots": self.num_dots,
            "size": self.dots_size,
        }

    def init_dots(self) -> dict:
        return {
            "colors": self._rng.random((self.num_dots, 3)),
            "sizes": self._rng.uniform(0.8, 1.2, size=(self.num_dots, 1)),
        }

    def reset(self, eval_mode: bool, seed=None):
        super().reset(seed)
        self.dots_parameters = self.init_dots()
        self.dots_state = self.dots_behaviour.init_state(self.num_dots, self.x_lim, self.y_lim, self._rng)

    def build_bg(self, w, h):
        bg = np.zeros((h, w, 3))
        positions = self.dots_behaviour.get_positions(self.dots_state) * [[w, h]]
        sizes = self.dots_parameters["sizes"]
        colors = self.dots_parameters["colors"]
        for position, size, color in zip(positions, sizes, colors):
            cv2.circle(
                bg,
                (int(position[0]), int(position[1])),
                int(size * w * self.dots_size),
                color,
                -1,
            )

        self.dots_state = self.dots_behaviour.update_state(self.dots_state)
        bg *= 255
        return bg.astype(np.uint8)

    def get_image(self):
        h, w = self.shape
        img = self.build_bg(w, h)
        mask = np.logical_or(img[:, :, 0] > 0, img[:, :, 1] > 0, img[:, :, 2] > 0)
        return img, mask


# # from multimodal_gym.utils.image_utils import to_tensor
# import torch
# def to_tensor(x, dtype=torch.float32):
#     return torch.tensor(x, dtype=dtype)

# def to_numpy(x, dtype=np.float32):
#     return x.numpy().astype(dtype)

# def to_cpu(x):
#     return x.to('cpu')

# def to_gpu(x):
#     return x.to('cuda')

# obs = cv2.imread("/home/elle/Videos/isaac_lab/render/545.png")
# obs = to_tensor(obs, dtype=torch.float32)


# if obs.device == 'cuda':
#     obs = to_cpu(obs)

# obs = to_numpy(obs, np.uint8)

# disc = DotsSource(seed=42, shape=(80,80), dots_behaviour="constant", num_dots=10)
# img, mask = disc.get_image()

# augmented_obs = np.copy(obs)
# augmented_obs[mask] = img[mask]

# cv2.imshow('Image', obs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Image', augmented_obs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
