import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        # action = None
        action: np.ndarray = self(ptu.from_numpy(obs).float())
        ### END EDIT ###

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        # if self.discrete:
        #     # TODO: define the forward pass for a policy with a discrete action space.
        #     pass
        # else:
        #     # TODO: define the forward pass for a policy with a continuous action space.
        #     pass
        ### BEGIN EDIT 3.1 ###
        logits = self.logits_net(obs)
        if self.discrete:
            # Sample randomly from the logits distribution.
            probabilities = F.softmax(logits, dim=-1)
            action = distributions.Categorical(probabilities).sample()
        else:
            # Model parameterizes Gaussian distribution, so we sample from THAT.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            dist = distributions.Normal(mean, std)
            action = dist.sample()
        
        action = ptu.to_numpy(action)
        ### END EDIT ###
        return action

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        # Results in a bit of numerical instability but... Whatever.
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # print(actions)
        # TODO: implement the policy gradient actor update.
        # loss = None
        ### BEGIN EDIT 3.1 ###
        self.optimizer.zero_grad()

        # First calculate action logits.
        logits = self.logits_net(obs)
        # Then calculate the log probabilities of the actions taken.
        if self.discrete:
            actions = actions.long()  # Ensure actions are long for indexing.
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            dist = distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)

        # Calculate gradient of objective (Log probs * advantages).
        # This loads the loss gradients into the policy parameters.
        # Since we are performing gradient ascent we should NEGATE the loss.
        loss = -(log_probs * advantages).mean()

        # Perform backprop step.
        loss.backward()
        self.optimizer.step()
        ### END EDIT ###

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
