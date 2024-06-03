from typing import List

import torch
import torch.nn as nn
import numpy as np

from baselines.PPGA.models.policy import StochasticPolicy
from typing import Union, Optional


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(StochasticPolicy):
    def __init__(self,
                 obs_shape: Union[int, tuple],
                 action_shape: np.ndarray,
                 normalize_obs: bool = False,
                 normalize_returns: bool = False):
        StochasticPolicy.__init__(self, normalize_obs=normalize_obs, obs_shape=obs_shape, normalize_returns=normalize_returns)

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, np.prod(action_shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def forward(self, x):
        return self.actor_mean(x)

    def get_action(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy()


class PGAMEActor(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, np.prod(action_shape)),
            nn.Tanh()
        )
        self.actor_logstd = -100.0 * torch.ones(action_shape[0])

    def forward(self, obs):
        return self.actor_mean(obs)

    def get_action(self, obs):
        return self.forward(obs)

    def serialize(self):
        '''
        Returns a 1D numpy array view of the entire policy.
        '''
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array: np.ndarray):
        '''
        Update the weights of this policy with the weights from the 1D
        array of parameters
        '''
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self


class CriticBase(nn.Module):
    def __init__(self):
        super().__init__()

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    # From DQDRL paper https://arxiv.org/pdf/2202.03666.pdf
    def serialize(self):
        '''
        Returns a 1D numpy array view of the entire policy.
        '''
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array: np.ndarray):
        '''
        Update the weights of this policy with the weights from the 1D
        array of parameters
        '''
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self

    def gradient(self):
        '''Returns 1D numpy array view of the gradients of this actor'''
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()]
        )


class Critic(CriticBase):
    def __init__(self, obs_shape):
        '''
        Standard critic used in PPO. Used to move the mean solution point
        '''
        CriticBase.__init__(self)
        self.core = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_value(self, obs):
        core_out = self.core(obs)
        return self.critic(core_out)

    def forward(self, obs):
        return self.get_value(obs)


class QDCritic(CriticBase):
    def __init__(self,
                 obs_shape: Union[int, tuple],
                 measure_dim: int,
                 critics_list: Optional[List[nn.Module]] = None):
        '''
        Instantiates a multi-headed critic used for objective-measure gradient estimation
        :param obs_shape: shape of the observation space
        :param measure_dim: number of measures
        :param critics_list: Use this to pass in existing pre-trained critics
        '''
        CriticBase.__init__(self)
        self.measure_dim = measure_dim
        if critics_list is None:
            self.all_critics = nn.ModuleList([
                nn.Sequential(
                    layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
                    nn.Tanh(),
                    layer_init(nn.Linear(256, 256)),
                    nn.Tanh(),
                    layer_init(nn.Linear(256, 1), std=1.0)
                ) for _ in range(measure_dim + 1)
            ])
        else:
            self.all_critics = nn.ModuleList(critics_list)

    def get_value_at(self, obs, dim):
        return self.all_critics[dim](obs)

    def get_all_values(self, obs):
        all_vals = []
        for critic in self.all_critics:
            all_vals.append(critic(obs))
        all_vals = torch.cat(all_vals).to(obs.device)
        return all_vals

    def get_value(self, obs):
        '''
        Implemented for backwards compatibility
        '''
        return self.all_critics[0](obs)
