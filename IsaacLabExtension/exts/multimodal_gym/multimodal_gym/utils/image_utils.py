import numpy as np
import torch
import os
import torchvision.transforms as transforms
from PIL import Image


def save_image(obs, img_name="lift.png", nchw=True):
    """
    Expects image in NCHW format, if in NHWC then set nchw=False
    """
    # filesaving
    img_dir = '/home/tmci/IsaacLab/IsaacLabExtension/exts/multimodal_gym/multimodal_gym/tasks/franka/'
    file_path = os.path.join(img_dir, img_name)

    print("input obs size:", img_name, obs.size(), obs.dtype)

    # reshape to have channels at last index
    if nchw:
        obs = obs.transpose(1, -1)
        print("transposing")

    print("save obs size:", obs.size(), obs.dtype)
    print("***********")

    nhwc_obs = obs
    nhwc_obs = np.array(nhwc_obs[0, :, :, :3].cpu() * 255).astype(np.uint8)
    # the image needs to be in the shape [height, width, channels]
    img = Image.fromarray(nhwc_obs)
    img.save(file_path)


def save_images_to_file(images: torch.Tensor, file_path: str):
    """Save images to file.

    Args:
        images: A tensor of shape (N, H, W, C) containing the images.
        file_path: The path to save the images to.

    Expects float32
    """
    from torchvision.utils import make_grid, save_image

    # make (N, C, H, W)
    # print(images.dtype)
    # print(f'IMAGES GRID: {images.shape}')
    # if images.dtype == torch.float32:
    #     images = images * 255
    #     images = images.to(torch.uint8)
    # print(f'{images.dtype} // [{images.min()}, {images.max()}]')

    # images = images[0,:,:,:]
    # images = images.cpu().reshape(images.shape[-1], images.shape[0], images.shape[1])
    # print(f'\n\n\n{images.shape}')
    #
    # to_pil = transforms.ToPILImage()
    #
    # image = to_pil(images)
    # print(f'\n\n\n{images.shape}')
    # # Save the image
    # image.save(file_path)
    # qqq


    grid = torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1)
    print(f'IMAGE NEW: {grid.shape}')
    save_image(make_grid(grid,
                         nrow=round(images.shape[0] ** 0.5)
                         ), file_path)


def save_stacked_image(images: torch.Tensor, file_path: str):

    # swap (N,H,W,C) to (N, C, H, W)
    stacked_frames = torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1)

    stacked_frames = stacked_frames[0, :, :, :]

    # Ensure the tensor is in the correct format (C, H, W) for torchvision
    # stacked_frames = stacked_frames.permute(2, 0, 1)  # Resulting shape: (9, 80, 80)
    print(stacked_frames.shape)
    # Convert to uint8 for image saving
    stacked_frames = stacked_frames - stacked_frames.min()
    stacked_frames = stacked_frames / stacked_frames.max() * 255
    stacked_frames = stacked_frames.to(torch.uint8)

    # Convert the tensor to a PIL image
    to_pil = transforms.ToPILImage()

    image = to_pil(stacked_frames)

    # Save the image
    image.save("stacked_frames_image.png")


