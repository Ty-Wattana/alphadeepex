import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from stable_baselines3 import DQN
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks

# Polyak update helper function
def polyak_update(source_params, target_params, tau):
    for src, tgt in zip(source_params, target_params):
        tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)

def adjust_action_mask(mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
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

class LSTMQNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, vocab_size=None, embedding_dim=None):
        """
        If vocab_size and embedding_dim are provided, the network will use an embedding layer
        to convert discrete token indices into dense vectors before processing.
        """
        super(LSTMQNet, self).__init__()
        self.hidden_dim = hidden_dim
        # If tokenized inputs are used, add an embedding layer.
        if vocab_size is not None and embedding_dim is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            fc_input_dim = embedding_dim
        else:
            self.embedding = None
            fc_input_dim = input_dim
        self.fc = nn.Linear(fc_input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_actions)

    def extract_features(self, obs):
        """
        If an embedding layer is defined and the observation is integer (token indices),
        embed the tokens. Otherwise, process as before.
        """
        # If obs is a token sequence (e.g. LongTensor) and embedding layer is defined.
        if self.embedding is not None and obs.dtype in [torch.long, torch.int]:
            # Expect obs shape to be (batch, seq_len)
            obs = self.embedding(obs)
        else:
            # If obs is 4D (batch, channels, height, width), flatten spatial dims
            if obs.dim() == 4:
                obs = obs.view(obs.size(0), -1)
                obs = obs.unsqueeze(1)
            # If obs is 5D (batch, seq_len, channels, height, width), flatten spatial dims
            elif obs.dim() == 5:
                batch, seq_len, c, h, w = obs.size()
                obs = obs.view(batch, seq_len, c * h * w)
            # Otherwise assume obs is already in (batch, seq_len, features) or (batch, features)
            # If obs is 2D (batch, features), add a sequence dimension.
            elif obs.dim() == 2:
                obs = obs.unsqueeze(1)
        return obs

    def forward(self, obs, hidden=None):
        # Extract features (using embedding if defined)
        features = self.extract_features(obs)
        features = features.float()
        features = F.relu(self.fc(features))
        # Flatten LSTM parameters to ensure they are contiguous in memory.
        self.lstm.flatten_parameters()
        # If no hidden state is passed, initialize with zeros.
        if hidden is None:
            batch = features.size(0)
            h0 = torch.zeros(1, batch, self.hidden_dim, device=features.device)
            c0 = torch.zeros(1, batch, self.hidden_dim, device=features.device)
            hidden = (h0, c0)
        # Process through LSTM.
        x, hidden = self.lstm(features, hidden)
        # Use the output of the last time step.
        x = x[:, -1, :]
        q_values = self.out(x)
        return q_values, hidden

class BootstrappedDQN(DQN):
    def __init__(
        self,
        policy: str,
        env,
        num_bootstrapped_nets: int = 10,
        mask_prob: float = 0.8,
        device="cuda",
        # Optionally pass embedding parameters if using token sequences.
        vocab_size: int = None,
        embedding_dim: int = None,
        **kwargs,
    ):
        super().__init__(policy, env, **kwargs)
        self.num_bootstrapped_nets = num_bootstrapped_nets
        self.mask_prob = mask_prob
        self.device = device

        # Get environment dimensions.
        # If using token sequences, input_dim should be set to the sequence length (or token feature dim)
        obs_dim = env.observation_space.shape[0]
        num_actions = env.action_space.n
        hidden_dim = 64  # arbitrary hidden dimension

        # Create independent copies of the Q-network and its target network for each head.
        self.q_networks = nn.ModuleList([
            copy.deepcopy(LSTMQNet(obs_dim, hidden_dim, num_actions, vocab_size, embedding_dim).to(device))
            for _ in range(num_bootstrapped_nets)
        ])
        self.target_q_networks = nn.ModuleList([
            copy.deepcopy(net) for net in self.q_networks
        ])
        # For simplicity we use a single optimizer over all network parameters.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # For deep exploration: current_head indicates which head to follow this episode.
        self.current_head = np.random.randint(0, self.num_bootstrapped_nets)

    def parameters(self):
        params = []
        for net in self.q_networks:
            params += list(net.parameters())
        return params

    def sample_episode_head(self):
        """Call this at the beginning of each episode to choose a head for deep exploration."""
        self.current_head = np.random.randint(0, self.num_bootstrapped_nets)

    def _sample_bootstrapped_masks(self, batch_size: int) -> torch.Tensor:
        masks = torch.rand((self.num_bootstrapped_nets, batch_size)) < self.mask_prob
        return masks.float().to(self.device)

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def train(self, gradient_steps=1, batch_size: int = 100):
        self.sample_episode_head()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        batch = self.replay_buffer.sample(batch_size)

        obs = batch.observations.to(self.device)
        actions = batch.actions.long().to(self.device)
        rewards = batch.rewards.float().to(self.device)
        next_obs = batch.next_observations.to(self.device)
        dones = batch.dones.float().to(self.device)

        reward_mean = torch.mean(rewards)
        self.logger.record("train/mean_reward", reward_mean.item())

        # Here, we assume each observation is already a token sequence.
        # (If using continuous observations, adjust accordingly.)
        hidden = None  # For training, you might consider preserving hidden state across tokens.

        masks = self._sample_bootstrapped_masks(batch_size)
        losses = []
        for i in range(self.num_bootstrapped_nets):
            with torch.no_grad():
                next_q, _ = self.target_q_networks[i](next_obs, hidden)
                next_max_q, _ = torch.max(next_q, dim=1)
                rewards_squeezed = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
                dones_squeezed = dones.squeeze(-1) if dones.dim() > 1 else dones
                target_q = rewards_squeezed + self.gamma * (1 - dones_squeezed) * next_max_q
                # target_q = rewards + self.gamma * (1 - dones) * next_max_q

            current_q, _ = self.q_networks[i](obs, hidden)
            if current_q.dim() > 2:
                current_q = current_q.view(current_q.size(0), -1)
            action_indices = actions.unsqueeze(1) if actions.dim() == 1 else actions
            current_q = current_q.gather(1, action_indices).squeeze(1)
            loss = F.mse_loss(current_q, target_q, reduction='none')
            loss = loss * masks[i].mean()
            losses.append(torch.mean(loss).item())
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.record("train/loss", np.mean(losses))
        for i in range(self.num_bootstrapped_nets):
            polyak_update(self.q_networks[i].parameters(),
                          self.target_q_networks[i].parameters(),
                          self.tau)

    def predict(self, observation, hidden=None, use_ensemble=False):
        # If observations are token sequences (e.g. a list or array of token indices),
        # we only add the batch dimension.
        if isinstance(observation, (list, np.ndarray)):
            obs = torch.tensor(observation, dtype=torch.long if self.q_networks[0].embedding is not None else torch.float32, device=self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
        else:
            obs = observation.to(self.device)
        with torch.no_grad():
            if use_ensemble:
                qs = []
                for net in self.q_networks:
                    q, hidden_out = net(obs, hidden)
                    qs.append(q)
                q_values = torch.stack(qs, dim=0).mean(dim=0)
            else:
                q_values, hidden_out = self.q_networks[self.current_head](obs, hidden)
        raw_mask = torch.as_tensor(get_action_masks(self.env)).to(self.device)
        mask = adjust_action_mask(raw_mask, q_values.shape)
        q_values = torch.where(mask, q_values, torch.tensor(-float('inf')).to(self.device))
        action = torch.argmax(q_values, dim=1).cpu().numpy()
        return action, hidden_out

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
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