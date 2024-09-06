import numpy as np
import torch

def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


def to_numpy(x, dtype=np.float32):
    return x.numpy().astype(dtype)


def to_cpu(x):
    return x.to("cpu")


def to_gpu(x):
    return x.to("cuda")
