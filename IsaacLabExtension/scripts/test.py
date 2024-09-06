import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from torchvision.utils import save_image
import numpy as np
import cv2
from multimodal_gym.utils.image_utils import *
from multimodal_gym.distractors.dots_source import DotsSource

discs = DotsSource(seed=42, shape=(80,80), dots_behaviour="constant")


# img_batch = np.load("img_batch_f32.npy")
# img_batch = (img_batch * 255).astype(np.uint8)

# # print(uint8_image[0, 40:45,40:45,:])
# cv2.imwrite(f"/home/elle/Videos/isaac_lab/render/0_obs.png", img_batch[0])


# obs = to_tensor(img_batch) 
# obs = to_gpu(obs) 

import torch

batch_size = 40
img_batch = torch.rand((batch_size, 80, 80, 3), dtype=torch.float32)


disc_img, disc_mask = discs.get_image()
disc_img_tensor = torch.tensor(disc_img, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
disc_mask_tensor = torch.tensor(disc_mask, dtype=torch.bool).unsqueeze(2).repeat(batch_size, 1, 1, 3)
img_batch[disc_mask_tensor] = disc_img_tensor[disc_mask_tensor]



print(np.shape(img_batch))

# disc_img = disc_img.astype(np.uint8)
# # print(uint8_image[40:45,40:45,:])

# disc_img = disc_img[np.newaxis, :, :, :]
# disc_mask = disc_mask[np.newaxis, :, :]

# masked_img_batch = np.copy(img_batch)
# masked_

# cv2.imwrite(f"/home/elle/Videos/isaac_lab/render/0_newobs.png", masked_img_batch[0])



# cv2.imwrite("/home/elle/Videos/isaac_lab/render/0_discimgf32.png", disc_img)
# obs = to_tensor(disc_img)

# # obs = to_gpu(obs)

# first_img = obs[0,:,:,:]
# print(np.shape(first_img))

# save_images_to_file(obs, f"/home/elle/Videos/isaac_lab/render/0_discimg_method.png")
# save_image(first_img, 'disc_img.png')

# print(disc_img.dtype)
# print(np.shape(disc_img), np.shape(disc_mask), np.shape(img_batch))

# obs = to_tensor(disc_img) 
# obs = to_gpu(obs)
# save_images_to_file(obs, f"/home/elle/Videos/isaac_lab/render/0_discimg.png")



# obs = to_tensor(masked_img_batch) 
# obs = to_gpu(obs) 

# save_images_to_file(obs, f"/home/elle/Videos/isaac_lab/render/0f32.png")
# exit()
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Image', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
