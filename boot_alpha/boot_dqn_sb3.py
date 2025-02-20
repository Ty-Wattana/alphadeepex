import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from stable_baselines3 import DQN
from typing import Optional
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.noise import ActionNoise
from sb3_contrib.common.maskable.utils import get_action_masks
from gymnasium import spaces
import torch.nn.functional as F

class BootstrappedDQN(DQN):
    def __init__(
        self,
        policy: str,
        env,
        num_bootstrapped_nets: int = 10,
        mask_prob: float = 0.8,
        **kwargs,
    ):
        super().__init__(policy, env, **kwargs)
        self.num_bootstrapped_nets = num_bootstrapped_nets
        self.mask_prob = mask_prob

        # Replace the single Q-network with a list of Q-networks
        self.q_networks = nn.ModuleList(
            [self.policy.q_net for _ in range(num_bootstrapped_nets)]
        )
        self.target_q_networks = nn.ModuleList(
            [self.policy.q_net_target for _ in range(num_bootstrapped_nets)]
        )

    def _sample_bootstrapped_masks(self, batch_size: int) -> torch.Tensor:
        """
        Generate bootstrapped masks for each Q-network.
        Each mask is a binary tensor of shape (batch_size,).
        """
        masks = torch.rand((self.num_bootstrapped_nets, batch_size)) < self.mask_prob
        return masks.float()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Override the train method to train each Q-network with bootstrapped masks.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        for _ in range(gradient_steps):
            # Sample a batch of transitions from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Generate bootstrapped masks for the batch
            masks = self._sample_bootstrapped_masks(batch_size).to(self.device)

            # Train each Q-network independently
            for i in range(self.num_bootstrapped_nets):
                with torch.no_grad():
                    # Compute the target Q-values using the target network
                    next_q_values = self.target_q_networks[i](replay_data.next_observations)
                    next_q_values, _ = torch.max(next_q_values, dim=1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Compute the current Q-values
                current_q_values = self.q_networks[i](replay_data.observations)
                current_q_values = current_q_values.gather(1, replay_data.actions.long()).squeeze(1)

                # Apply the bootstrapped mask
                masked_loss = masks[i] * nn.functional.mse_loss(current_q_values, target_q_values)

                # Optimize the Q-network
                self.policy.optimizer.zero_grad()
                masked_loss.mean().backward()
                self.policy.optimizer.step()

        # Update target networks
        self._update_target_network()

    def _update_target_network(self) -> None:
        """
        Update the target networks for all bootstrapped Q-networks.
        """
        for i in range(self.num_bootstrapped_nets):
            polyak_update(
                self.q_networks[i].parameters(),
                self.target_q_networks[i].parameters(),
                self.tau,
            )

    def predict(self, 
                observation: np.ndarray, 
                state: Optional[np.ndarray] = None, 
                use_masking: bool = True,
                **kwargs) -> np.ndarray:
        """
        Override the predict method to use the ensemble of Q-networks.
        """

        observation = np.array(observation)
        with torch.no_grad():
            observation_tensor = torch.as_tensor(observation).to(self.device)

            # If the policy is stateful (e.g., RNN), pass the state through the network
            if hasattr(self.policy, 'is_recurrent') and self.policy.is_recurrent:
                q_values, state = self.policy.q_net(observation_tensor, state)
                q_values = torch.stack([q_net(observation_tensor, state) for q_net in self.q_networks])
            else:
                q_values = torch.stack([q_net(observation_tensor) for q_net in self.q_networks])
            
            # Average Q-values across all networks
            q_values = q_values.mean(dim=0)

            if use_masking:
                action_masks = torch.as_tensor(get_action_masks(self.env)).to(self.device)
                # Set Q-values of invalid actions to negative infinity
                q_values = torch.where(action_masks, q_values, torch.tensor(-float('inf')).to(self.device))

            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        return actions, state    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Override the _sample_action method to mask epsilon greedy action selection.
        """
        
        if self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate:
            # Sample random action
            action_masks = torch.as_tensor(get_action_masks(self.env)).cpu().numpy()
            valid_actions = [np.where(action_masks[i])[0] for i in range(n_envs)]
            unscaled_action = np.array([np.random.choice(valid_actions[i]) for i in range(n_envs)])
        else:
            # Use the ensemble's predicted action
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
class MaskDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(MaskDQN, self).__init__(*args, **kwargs)

    def _apply_action_mask(self, q_values, action_mask):
        """
        Applies the action mask to the q_values tensor.
        
        Parameters:
            q_values (torch.Tensor): Q-values of shape (batch_size, n_actions)
            action_mask (np.ndarray): Boolean array of shape (n_actions,) or (batch_size, n_actions)
                                       indicating which actions are valid (True) or invalid (False).
                                       
        Returns:
            torch.Tensor: Masked q_values with invalid actions set to -infinity.
        """
        # Convert the numpy mask to a torch tensor on the same device as q_values
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device)
        # If the mask is 1D, expand it to match the batch dimension
        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).expand_as(q_values)
        # Clone q_values to avoid modifying the original tensor
        q_values = q_values.clone()
        # Set Q-values for invalid actions to -infinity
        q_values[~mask_tensor] = -float('inf')
        return q_values

    def predict(self, observation, state=None, deterministic=False):
        """
        Predict the action given an observation while masking out invalid actions.
        It uses the environment's get_action_masks method if available.
        
        Parameters:
            observation: The observation from the environment.
            state: RNN state (if applicable).
            deterministic (bool): Whether to use deterministic actions.
        
        Returns:
            action (np.ndarray): The selected action.
            state: The (unchanged) RNN state.
        """
        # Get the valid action mask from the environment, if available.
        # The mask should be a boolean array of shape (n_actions,) where True indicates a valid action.
        if hasattr(self.env, "get_action_masks"):
            action_mask = torch.as_tensor(get_action_masks(self.env)).to(self.device)
        else:
            # If no mask is provided, assume all actions are valid.
            action_mask = np.ones(self.env.action_space.n, dtype=bool)
        
        # Convert observation to tensor (the policy helper does this conversion)
        obs_tensor, _ = self.policy.obs_to_tensor(observation)
        
        with torch.no_grad():
            # Compute Q-values from the policy's Q-network
            q_values = self.policy.q_net(obs_tensor)
            # Apply the action mask: invalid actions get a Q-value of -infinity.
            masked_q_values = self._apply_action_mask(q_values, action_mask)
        
        if deterministic:
            # Choose the action with the highest masked Q-value.
            actions = torch.argmax(masked_q_values, dim=1)
        else:
            # Alternatively, sample from the softmax probability distribution over masked Q-values.
            probs = F.softmax(masked_q_values, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return actions.cpu().numpy(), state