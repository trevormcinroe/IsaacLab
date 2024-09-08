from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from enum import Enum
import gym
import gymnasium

import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import Model  # noqa
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, MultivariateGaussianMixin  # noqa

from skrl.utils.model_instantiators.torch import _generate_sequential, _get_activation_function, _get_num_units_by_shape

from multimodal_gym.utils.frame_stack import LazyFrames

class Shape(Enum):
    """
    Enum to select the shape of the model's inputs and outputs
    """
    ONE = 1
    STATES = 0
    OBSERVATIONS = 0
    ACTIONS = -1
    STATES_ACTIONS = -2

def custom_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 structure: str = "",
                 roles: Sequence[str] = [],
                 parameters: Sequence[Mapping[str, Any]] = [],
                 single_forward_pass: bool = True,
                 frame_stack: int = 1, num_gt_observations: int = 4) -> Model:
    """Instantiate a shared model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param structure: Shared model structure (default: ``""``).
                      Note: this parameter is ignored for the moment
    :type structure: str, optional
    :param roles: Organized list of model roles (default: ``[]``)
    :type roles: sequence of strings, optional
    :param parameters: Organized list of model instantiator parameters (default: ``[]``)
    :type parameters: sequence of dict, optional
    :param single_forward_pass: Whether to perform a single forward-pass for the shared layers/network (default: ``True``)
    :type single_forward_pass: bool

    :return: Shared model instance
    :rtype: Model
    """
    class GaussianDeterministicModel(GaussianMixin, DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, roles, metadata, single_forward_pass, frame_stack, num_gt_observations):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self,
                                   clip_actions=metadata[0]["clip_actions"],
                                   clip_log_std=metadata[0]["clip_log_std"],
                                   min_log_std=metadata[0]["min_log_std"],
                                   max_log_std=metadata[0]["max_log_std"],
                                   role=roles[0])
            DeterministicMixin.__init__(self, clip_actions=metadata[1]["clip_actions"], role=roles[1])

            self._roles = roles
            self._single_forward_pass = single_forward_pass
            self.instantiator_input_type = metadata[0]["input_shape"].value
            self.instantiator_output_scales = [m["output_scale"] for m in metadata]

            print("Observation space:", observation_space)
            num_inputs = observation_space.shape[0]
            num_actions = metadata[1]["output_shape"].value

            self.obs_type = metadata[0]["obs_type"]

            # print(observation_space)
            self.num_gt_observations = num_gt_observations
            
            num_inputs, self.cnn = process_inputs(self.obs_type, frame_stack, metadata[0]["latent_dim"], metadata[0]["img_dim"], num_inputs, num_gt_observations)

            # shared layers/network
            self.net = nn.Sequential(nn.Linear(num_inputs, 32),
                                nn.ELU(),
                                nn.Linear(32, 32),
                                nn.ELU()).to(device)
            
            self.mean_net = nn.Sequential(nn.Linear(32, num_actions), 
                                            nn.Tanh()).to(device)
            self.log_std_parameter = nn.Parameter(torch.zeros(num_actions)).to(device)

            self.value_net = nn.Sequential(nn.Linear(32, 1),
                                             nn.Identity()).to(device)

        def act(self, inputs, role):
            print(f'WITHIN act()!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            qqq
            if role == self._roles[0]:
                return GaussianMixin.act(self, inputs, role)
            elif role == self._roles[1]:
                return DeterministicMixin.act(self, inputs, role)

        def compute(self, inputs, role):
            if self.instantiator_input_type == 0:
                net_inputs = inputs["states"]
            elif self.instantiator_input_type == -1:
                net_inputs = inputs["taken_actions"]
            elif self.instantiator_input_type == -2:
                net_inputs = torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)

            if self.obs_type == "image":
                # pass input first through cnn
                net_inputs = self.cnn(net_inputs)

            elif self.obs_type == "concat":
                # pass input first through cnn
                prop_obs = net_inputs[:, :self.num_gt_observations]
                img_obs = net_inputs[:, self.num_gt_observations:]
                z = self.cnn(img_obs)
                net_inputs = torch.cat((prop_obs, z), dim=1)

            # single shared layers/network forward-pass
            if self._single_forward_pass:
                if role == self._roles[0]:
                    self._shared_output = self.net(net_inputs)
                    return self.instantiator_output_scales[0] * self.mean_net(self._shared_output), self.log_std_parameter, {}
                elif role == self._roles[1]:
                    shared_output = self.net(net_inputs) if self._shared_output is None else self._shared_output
                    self._shared_output = None
                    return self.instantiator_output_scales[1] * self.value_net(shared_output), {}
            # multiple shared layers/network forward-pass
            else:
                raise NotImplementedError

    # TODO: define the model using the specified structure

    return GaussianDeterministicModel(observation_space=observation_space,
                                      action_space=action_space,
                                      device=device,
                                      roles=roles,
                                      metadata=parameters,
                                      single_forward_pass=single_forward_pass,
                                      frame_stack=frame_stack, 
                                      num_gt_observations=num_gt_observations)

def process_inputs(obs_type, frame_stack, latent_dim, img_dim, num_inputs, num_gt_observations):
            
    # create cnn if image included
    print("obs type is ", obs_type)
    if obs_type == "image" or obs_type == "concat":
        obs_dim = (frame_stack*3, img_dim, img_dim)
        cnn = ImageEncoder(obs_dim, img_dim, frame_stack, feature_dim=latent_dim, num_layers=2)
    else:
        cnn = None

    if obs_type == "image":
        num_inputs = latent_dim
    elif obs_type == "concat":
        num_inputs = latent_dim + num_gt_observations        


    return num_inputs, cnn

def custom_gaussian_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   device: Optional[Union[str, torch.device]] = None,
                   clip_actions: bool = False,
                   clip_log_std: bool = True,
                   min_log_std: float = -20,
                   max_log_std: float = 2,
                   initial_log_std: float = 0,
                   input_shape: Shape = Shape.STATES,
                   hiddens: list = [256, 256],
                   hidden_activation: list = ["relu", "relu"],
                   output_shape: Shape = Shape.ACTIONS,
                   output_activation: Optional[str] = "tanh",
                   output_scale: float = 1.0,
                   obs_type: str = "prop",
                   latent_dim: int = 512,
                   img_dim: int = 84,
                   frame_stack: int = 1, num_gt_observations: int = 4) -> Model:
    """Instantiate a Gaussian model

    :return: Gaussian model instance
    :rtype: Model
    """

    class GaussianModel(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions,
                     clip_log_std, min_log_std, max_log_std, frame_stack, num_gt_observations, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            self.obs_type = obs_type

            num_inputs = observation_space.shape[0]
            num_actions = action_space.shape[0]
            self.num_gt_observations = num_gt_observations
            # print("Policy: num inputs, actions, gt:", num_inputs, num_actions, self.num_gt_observations)
            #
            # print(f'img_dim: {img_dim} // fstack: {frame_stack}')
            # qqq

            num_inputs, self.cnn = process_inputs(obs_type, frame_stack, latent_dim, img_dim, num_inputs, num_gt_observations)
            self.cnn = self.cnn.to(device)
            # shared layers/network
            self.net = nn.Sequential(nn.Linear(num_inputs, 256),
                                nn.ELU(),
                                nn.Linear(256, 128),
                                nn.ELU(),
                                nn.Linear(128, 64),
                                nn.ELU(),
                                nn.Linear(64, num_actions), 
                                nn.Tanh()).to(device)
            
            print(self.cnn, self.net)
            
            self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(num_actions)).to(device)



        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                net_inputs = inputs["states"]
            elif self.instantiator_input_type == -1:
                net_inputs = inputs["taken_actions"]
            elif self.instantiator_input_type == -2:
                net_inputs = torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)

            if self.obs_type == "image":
                # pass input first through cnn
                net_inputs = self.cnn(net_inputs)
                # print(f'net_inputs: {net_inputs.shape}')

            elif self.obs_type == "concat":
                # pass input first through cnn
                prop_obs = net_inputs[:, :self.num_gt_observations]
                img_obs = net_inputs[:, self.num_gt_observations:]
                z = self.cnn(img_obs)
                net_inputs = torch.cat((prop_obs, z), dim=1)

            output = self.net(net_inputs)
            # print(f'output: {output.shape}')

            return output * self.instantiator_output_scale, self.log_std_parameter, {}

    metadata = {"input_shape": input_shape,
                "hiddens": hiddens,
                "hidden_activation": hidden_activation,
                "output_shape": output_shape,
                "output_activation": output_activation,
                "output_scale": output_scale,
                "initial_log_std": initial_log_std}

    return GaussianModel(observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         clip_actions=clip_actions,
                         clip_log_std=clip_log_std,
                         min_log_std=min_log_std,
                         max_log_std=max_log_std,
                         frame_stack=frame_stack,
                         num_gt_observations=num_gt_observations)

def custom_deterministic_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        clip_actions: bool = False,
                        input_shape: Shape = Shape.STATES,
                        hiddens: list = [256, 256],
                        hidden_activation: list = ["relu", "relu"],
                        output_shape: Shape = Shape.ACTIONS,
                        output_activation: Optional[str] = "tanh",
                        output_scale: float = 1.0,
                        obs_type: str = "prop",
                        latent_dim: int = 512, 
                        img_dim: int = 80,
                        frame_stack: int = 1, num_gt_observations: int = 4) -> Model:
    """Instantiate a deterministic model

    :return: Deterministic model instance
    :rtype: Model
    """
    class DeterministicModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions, frame_stack, num_gt_observations):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            self.obs_type = obs_type

            num_inputs = observation_space.shape[0]
            num_actions = action_space.shape[0]
            self.num_gt_observations = num_gt_observations
            print("num inputs, actions, gt:", num_inputs, num_actions, self.num_gt_observations)
            
            num_inputs, self.cnn = process_inputs(obs_type, frame_stack, latent_dim, img_dim, num_inputs, num_gt_observations)
            self.cnn = self.cnn.to(device)
            # shared layers/network
            self.net = nn.Sequential(nn.Linear(num_inputs, 256),
                                nn.ELU(),
                                nn.Linear(256, 128),
                                nn.ELU(),
                                nn.Linear(128, 64),
                                nn.ELU(),
                                nn.Linear(64, 1),
                                nn.Identity()).to(device)


        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                net_inputs = inputs["states"]
            elif self.instantiator_input_type == -1:
                net_inputs = inputs["taken_actions"]
            elif self.instantiator_input_type == -2:
                net_inputs = torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)

            if self.obs_type == "image":
                # pass input first through cnn
                net_inputs = self.cnn(net_inputs)
                # print(f'net_inputs: {net_inputs.shape}')
                # qqq

            elif self.obs_type == "concat":
                # pass input first through cnn
                prop_obs = net_inputs[:, :self.num_gt_observations]
                img_obs = net_inputs[:, self.num_gt_observations:]
                z = self.cnn(img_obs)
                net_inputs = torch.cat((prop_obs, z), dim=1)

            output = self.net(net_inputs)
            # print(f'output: {output.shape}')

            return output * self.instantiator_output_scale, {}

    metadata = {"input_shape": input_shape,
                "hiddens": hiddens,
                "hidden_activation": hidden_activation,
                "output_shape": output_shape,
                "output_activation": output_activation,
                "output_scale": output_scale}

    return DeterministicModel(observation_space=observation_space,
                              action_space=action_space,
                              device=device,
                              clip_actions=clip_actions,
                              frame_stack=frame_stack,
                              num_gt_observations=num_gt_observations)

class AE(nn.Module):
    def __init__(self, input_size, hidden_size=512, feature_dim=50):
        super().__init__()
        print("Initialising proprioception encoder")
        # Following architecture by You & Liu - 2 FCNs with sizes 512, 50
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, feature_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )
        self.enc_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def reconstruct(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class VAE(nn.Module):

    def __init__(self, input_size=784, hidden_size=400, feature_dim=200):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, feature_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )
        self.enc_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

        # latent mean and variance 
        self.mean_layer = nn.Linear(feature_dim, 2)
        self.logvar_layer = nn.Linear(feature_dim, 2)  
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def reconstruct(self, x):
        z = self.forward(x)
        x_hat = self.decode(z)
        return x_hat
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        return z



class ImageEncoder(nn.Module):
    """Convolutional encoder for image-based observations.

    You (2024) construact an image observation by stacking 3 frames,
    where each frame is 84x84x3. We divide each pixel by 255 and scale 
    it down to [0,1]. Before we feed images into the encoders, we follow
    Yarats 2021 data augmentation by random shift [-4, 4]
    
    obs_shape (C, W, H)

    nn.Conv2d(in_channels, out_channels, kernel_size)
    """
    def __init__(self, obs_shape, img_dim, frame_stack, feature_dim=50, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3
        num_layers = 4

        print("initialising cnn with ", num_layers, "and ", feature_dim, "feature dim")

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.output_logits = False
        self.feature_dim = feature_dim

        self.num_channels = obs_shape[0]
        self.img_dim = img_dim
        self.frame_stack = frame_stack
        
        kernel_size = 3

        self.convs = nn.ModuleList([nn.Conv2d(self.num_channels, self.num_filters, kernel_size=kernel_size, stride=2)])
        for i in range(1, self.num_layers):
            self.convs.extend([nn.Conv2d(self.num_filters, self.num_filters, kernel_size=kernel_size, stride=1)])

        # get output shape
        x = torch.randn(*obs_shape).unsqueeze(0)
        conv = torch.relu(self.convs[0](x))
        for i in range(1, self.num_layers):
            conv = self.convs[i](conv)
        conv = conv.view(conv.size(0), -1)
        self.output_shape = conv.shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.output_shape, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.out_dim = self.feature_dim

        self.apply(weight_init)

        self.outputs = dict()

    def forward_conv(self, obs):
        # print(f'obs in: {obs.shape} // {obs.dtype} [{obs.min()}, {obs.max()}] // {obs.device}')
        # obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        # sac-ae
        # h = conv.view(conv.size(0), -1)
        # check if mhairi's version makes a difference
        h = conv.reshape(conv.size(0), -1)
        return h

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        # obs coming in: torch.Size([4, 21168])
        # print(f'obs coming fin: {obs.shape}')
        # input obs is[N, H, W, C], expected to be [N, C, H, W]
        if isinstance(obs, LazyFrames):
            raise ValueError("CNN input is LazyFrame. Convert to tensor")
            print("converting lazy")
            # Concatenate along the last dimension from list of torch.Size([128, 80, 80, 3]) to ([128, 80, 80, 12])
            frames = obs._frames
            obs = torch.cat(frames, dim=-1)

        # print("cnn", obs.size())

        # obs is batch size x len
        batch_size = obs.size()[0]

        # the data that comes out of the replay buffer is corrupted with the following view...
        obs = obs.view(batch_size, self.img_dim, self.img_dim, self.num_channels)

        from multimodal_gym.utils.image_utils import save_images_to_file
        #
        print(f'obs: {obs.shape}')
        file_path = '/home/tmci/IsaacLab/IsaacLabExtension/exts/multimodal_gym/multimodal_gym/tasks/franka/lift.png'
        import numpy as np
        from PIL import Image
        obs = np.array(obs[0, :, :, :3].cpu() * 255).astype(np.uint8)
        img = Image.fromarray(obs)
        img.save(file_path)
        # save_images_to_file(
        #     next_obs.reshape(args_cli.num_envs, args_cli.hw, args_cli.hw, 3*args_cli.frame_stack)[:, :, :, :3],
        #     file_path)
        qqq
        # qqq
        # print("cnn", obs.size())

        h = self.forward_conv(obs)

        if detach_encoder_conv:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        if detach_encoder_head:
            out = out.detach()
        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def copy_head_weights_from(self, source):
        """Tie head layers"""
        for i in range(2):
            tie_weights(src=source.head[i], trg=self.head[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)



# https://github.com/denisyarats/pytorch_sac_ae/blob/master/encoder.py
def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py
#  mhairi - https://github.com/uoe-agents/MVD/blob/main/utils.py
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)