import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Optional, Tuple, List
from stable_baselines3 import DQN
from typing import Optional
# from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.noise import ActionNoise
from sb3_contrib.common.maskable.utils import get_action_masks
from gymnasium import spaces
import torch.nn.functional as F

# Polyak update helper function
def polyak_update(source_params, target_params, tau):
    for src, tgt in zip(source_params, target_params):
        tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)

def adjust_action_mask(mask: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Adjusts the action mask to have the same shape as target_shape (batch_size, num_actions).
    If the mask has fewer columns, it pads with zeros; if it has more, it truncates.
    """
    batch_size, n_actions_target = target_shape
    batch_size_mask, n_actions_mask = mask.shape
    if n_actions_mask < n_actions_target:
        padding = torch.zeros((batch_size, n_actions_target - n_actions_mask), dtype=mask.dtype, device=mask.device)
        adjusted_mask = torch.cat([mask, padding], dim=1)
    elif n_actions_mask > n_actions_target:
        adjusted_mask = mask[:, :n_actions_target]
    else:
        adjusted_mask = mask
    return adjusted_mask

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

        # Create independent copies of the Q-network and its target network for each head.
        self.q_networks = nn.ModuleList([
            copy.deepcopy(self.policy.q_net) for _ in range(num_bootstrapped_nets)
        ])
        self.target_q_networks = nn.ModuleList([
            copy.deepcopy(self.policy.q_net_target) for _ in range(num_bootstrapped_nets)
        ])
        
        # For deep exploration: current_head indicates which head to follow this episode.
        self.current_head = np.random.randint(0, self.num_bootstrapped_nets)
        
    def sample_episode_head(self):
        """Call this at the beginning of each episode to choose a head for deep exploration."""
        self.current_head = np.random.randint(0, self.num_bootstrapped_nets)
    
    def _sample_bootstrapped_masks(self, batch_size: int) -> torch.Tensor:
        """
        Generate bootstrapped masks for each Q-network.
        Each mask is a binary tensor of shape (batch_size,).
        """
        masks = torch.rand((self.num_bootstrapped_nets, batch_size)) < self.mask_prob
        return masks.float()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train each Q-network using bootstrapped masks.
        Here we also sample a new head at the start of training (simulating an episode).
        In your full training loop, you should call sample_episode_head() whenever you reset your environment.
        """
        self.sample_episode_head()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            masks = self._sample_bootstrapped_masks(batch_size).to(self.device)

            for i in range(self.num_bootstrapped_nets):
                with torch.no_grad():
                    next_q_values = self.target_q_networks[i](replay_data.next_observations)
                    next_q_values, _ = torch.max(next_q_values, dim=1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                current_q_values = self.q_networks[i](replay_data.observations)

                # Ensure the output is 2D: (batch_size, num_actions)
                if current_q_values.dim() > 2:
                    current_q_values = current_q_values.view(current_q_values.size(0), -1)
                
                if replay_data.actions.dim() == 1:
                    action_indices = replay_data.actions.long().unsqueeze(1)
                else:
                    action_indices = replay_data.actions.long()
                
                if action_indices.dim() == 1:
                    action_indices = action_indices.unsqueeze(1)
                # Gather the Q-value for the taken actions.
                selected_q_values = current_q_values.gather(1, action_indices).squeeze(1)
                
                # Compute elementwise loss
                loss = F.mse_loss(selected_q_values, target_q_values, reduction='none')
                masked_loss = loss * masks[i]
                effective_batch_size = masks[i].sum() + 1e-6
                final_loss = masked_loss.sum() / effective_batch_size

                self.policy.optimizer.zero_grad()
                final_loss.backward()
                self.policy.optimizer.step()

        self._update_target_network()

    def _update_target_network(self) -> None:
        """
        Update the target networks using Polyak averaging.
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
                use_ensemble: bool = False,
                **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict actions using the Q-networks.
        If use_ensemble is False, the method uses the current episode’s head (for deep exploration).
        If use_ensemble is True, it averages the predictions of all heads.
        """
        observation = np.array(observation)
        with torch.no_grad():
            observation_tensor = torch.as_tensor(observation).to(self.device)
            if hasattr(self.policy, 'is_recurrent') and self.policy.is_recurrent:
                head_net = self.q_networks[self.current_head]
                if hasattr(head_net, 'rnn'):
                    head_net.rnn.flatten_parameters()
                q_values, state = head_net(observation_tensor, state)
            else:
                if use_ensemble:
                    q_values = torch.stack([q_net(observation_tensor) for q_net in self.q_networks])
                    q_values = q_values.mean(dim=0)
                else:
                    q_values = self.q_networks[self.current_head](observation_tensor)
            
            # If action masking is enabled, adjust the mask to match q_values' shape.
            # if kwargs.get('use_masking', False):
            raw_mask = torch.as_tensor(get_action_masks(self.env)).to(self.device)
            mask = adjust_action_mask(raw_mask, q_values.shape)
            q_values = torch.where(mask, q_values, torch.tensor(-float('inf')).to(self.device))
            
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions, state    

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action using epsilon–greedy exploration.
        """
        if self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate:
            action_masks = torch.as_tensor(get_action_masks(self.env)).cpu().numpy()
            valid_actions = [np.where(action_masks[i])[0] for i in range(n_envs)]
            unscaled_action = np.array([np.random.choice(valid_actions[i]) for i in range(n_envs)])
        else:
            unscaled_action, _ = self.predict(self._last_obs, use_ensemble=False)
        
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
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