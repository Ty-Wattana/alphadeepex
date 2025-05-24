import math
import random
import copy
import time
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.dqn.policies import MlpPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from collections import deque

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

# --- Helper: dynamic valid actions ---

def clone_env_state(env: Any) -> Any:
    # If the environment stores its state in a NumPy array attribute '_state'
    if hasattr(env, '_state'):
        # Use np.copy which is optimized in C
        return np.copy(env._state)
    # Otherwise fall back to using the provided get_state() method.
    return env.get_state()

def restore_env_state(env: Any, state: Any) -> None:
    if hasattr(env, '_state'):
        # If possible, update the state's content in-place.
        env._state[:] = state
    else:
        env.set_state(state)


def get_legal_actions(obs: Any, action_mask: Optional[List[bool]], action_space: spaces.Space) -> List[int]:
    """
    Returns the list of valid action indices.
    If an explicit action_mask is provided (a list of booleans), use it.
    Otherwise, if obs is a dict with an "action_mask" key, use that.
    Otherwise, fallback to all actions.
    """
    if action_mask is not None:
        mask = list(action_mask) if isinstance(action_mask, np.ndarray) else action_mask
        return [i for i, valid in enumerate(mask) if valid]
    if isinstance(obs, dict) and "action_mask" in obs:
        mask = obs["action_mask"]
        return [i for i, valid in enumerate(mask) if valid]
    if isinstance(action_space, spaces.Discrete):
        return list(range(action_space.n))
    else:
        raise NotImplementedError("Only discrete action spaces are supported.")
    
def compute_mcts_losses(batch, prior_net, gamma):
    """
    batch: list of tuples (obs, value_target, var_target)
    prior_net: your EpistemicRiskSeekerPolicy instance
    gamma: discount factor (unused here, but you might incorporate multi-step)
    """
    # unpack batch
    obs_list, value_targets, var_targets = zip(*batch)
    # move to tensors
    # prepare model inputs
    arr = np.stack([o if not isinstance(o, dict) else o['obs'] for o in obs_list])
    if arr.ndim == 2:
        arr = arr[:, None, :]
    x = torch.tensor(arr, dtype=torch.float32, device=prior_net.device)
    # forward pass once
    feat = prior_net.forward(x)        # (batch, hidden_dim)
    v_pred = prior_net.value_head(feat).squeeze(-1)       # shape (batch,)
    u_pred = prior_net.uncertainty_head(feat).squeeze(-1) # shape (batch,)
    # build targets
    v_target = torch.tensor(value_targets, dtype=torch.float32, device=prior_net.device)
    u_target = torch.tensor(var_targets,   dtype=torch.float32, device=prior_net.device)
    # losses
    loss_v = F.mse_loss(v_pred, v_target)
    loss_u = F.mse_loss(u_pred, u_target)
    return loss_v + loss_u

# --- EpistemicRiskSeekerPolicy ---
class EpistemicRiskSeekerPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 action_dim: int,
                 mlp_hidden_sizes: List[int] = [32, 32],
                 lr: float = 0.001,
                 device: Optional[str] = None):
        super(EpistemicRiskSeekerPolicy, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device is None else th.device(device)
        # Core representation
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True).to(self.device)
        # Prior head (action probabilities)
        layers = []
        last_dim = hidden_dim
        for size in mlp_hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, action_dim))
        self.prior_mlp = nn.Sequential(*layers).to(self.device)
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        ).to(self.device)
        # Optimizer for all parameters
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # shared GRU representation
        # x: (batch, seq_len, input_dim)
        _, h_n = self.gru(x)
        feat = h_n[-1]  # (batch, hidden_dim)
        return feat

    def get_prior(self, observation: Any) -> List[float]:
        # single observation to prior probabilities
        obs = observation
        if isinstance(obs, np.ndarray) and obs.ndim == 1:
            obs = obs.reshape(1, 1, -1)
        x = th.tensor(obs, dtype=th.float32, device=self.device)
        feat = self.forward(x)
        logits = self.prior_mlp(feat)
        probs = th.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        return probs.tolist()

    def get_prior_batch(self, observations: np.ndarray) -> List[List[float]]:
        # batch of observations
        obs = observations
        if obs.ndim == 2:
            obs = obs[:, None, :]
        x = th.tensor(obs, dtype=th.float32, device=self.device)
        feat = self.forward(x)
        logits = self.prior_mlp(feat)
        priors = th.softmax(logits, dim=-1).detach().cpu().numpy()
        return priors.tolist()

    def get_value(self, observations: List[Any]) -> List[float]:
        # returns value estimates for each observation
        batch = np.stack([o if not isinstance(o, dict) else o.get('obs', o) for o in observations])
        if batch.ndim == 2:
            batch = batch[:, None, :]
        x = th.tensor(batch, dtype=th.float32, device=self.device)
        feat = self.forward(x)
        vals = self.value_head(feat).detach().cpu().numpy().flatten().tolist()
        return vals

    def get_uncertainty(self, observations: List[Any]) -> List[float]:
        # returns positive uncertainty estimates
        batch = np.stack([o if not isinstance(o, dict) else o.get('obs', o) for o in observations])
        if batch.ndim == 2:
            batch = batch[:, None, :]
        x = th.tensor(batch, dtype=th.float32, device=self.device)
        feat = self.forward(x)
        uncs = self.uncertainty_head(feat).detach().cpu().numpy().flatten().tolist()
        return uncs

# --- Epistemic MCTS Node ---
class EpistemicMCTSNode:
    def __init__(self,
                 state: Any,
                 obs: Any,
                 parent: Optional['EpistemicMCTSNode']=None,
                 action: Optional[int]=None,
                 mask: Optional[List[bool]]=None,
                 prior: float=0.0):
        self.state = state
        self.obs = obs
        self.parent = parent
        self.action = action
        self.children: Dict[int, 'EpistemicMCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        self.sigma = 0.0
        self.done = False
        self.mask = mask
        self.prior = prior
        self._init_untried()

    def _init_untried(self):
        self.untried = get_legal_actions(self.obs, self.mask, self._dummy_space())

    def _dummy_space(self):
        if self.mask is not None:
            return spaces.Discrete(len(self.mask))
        if isinstance(self.obs, dict) and 'action_mask' in self.obs:
            return spaces.Discrete(len(self.obs['action_mask']))
        return spaces.Discrete(self.obs.shape[-1] if isinstance(self.obs, np.ndarray) else 1)

    def is_fully_expanded(self, alg: 'EpistemicMCTS') -> bool:
        legal = get_legal_actions(self.obs, self.mask, alg.action_space)
        return len(self.children) >= len(legal)

    def expand(self, alg: 'EpistemicMCTS',env) -> 'EpistemicMCTSNode':
        # Refresh legal actions under current mask
        legal = get_legal_actions(self.obs, self.mask, alg.action_space)
        # Exclude children already expanded
        untried = [a for a in legal if a not in self.children]
        if not untried:
            return self
        # Choose and remove one action
        action = random.choice(untried)
        # Simulate environment step
        new_state, new_obs, reward, done, info = alg.simulate(self.state, action,env)
        # Clone env to extract next mask
        temp_env = copy.deepcopy(env)
        restore_env_state(temp_env, new_state)
        child_mask = temp_env.action_masks() if hasattr(temp_env, 'action_masks') else None
        # Obtain prior probability from the prior network
        prior = alg.prior_net.get_prior(new_obs)[action]
        # Build child node
        child = EpistemicMCTSNode(new_state, new_obs, parent=self,
                                   action=action, mask=child_mask, prior=prior)
        child.done = done
        self.children[action] = child
        return child

    def best_child(self, alg: 'EpistemicMCTS', c: float) -> 'EpistemicMCTSNode':
        total = sum(ch.visits for ch in self.children.values())
        best, best_score = None, -float('inf')
        for ch in self.children.values():
            if ch.visits == 0:
                score = float('inf')
            else:
                exploit = ch.value / ch.visits
                explore = c * (ch.prior * math.sqrt(total) / (1 + ch.visits))
                # paper uses per-node average sigma w/o sqrt
                uncert = alg.beta * ch.sigma
                score = exploit + explore + uncert
            if score > best_score:
                best, best_score = ch, score
        return best

    def backup(self, reward: float, var: float):
        self.visits += 1
        self.value += reward
        delta = math.sqrt(var) - self.sigma
        self.sigma += delta / self.visits
        if self.parent:
            self.parent.backup(reward, var)

# --- Epistemic MCTS Algorithm ---
class EpistemicMCTS(BaseAlgorithm):
    def __init__(self,
                 env: gym.Env,
                 policy_net_kwargs: Dict[str, Any],
                 n_sims: int=100,
                 c: float=1.0,
                 beta: float=1.0,
                 gamma: float=1.0,
                 replay_size: int=10000,
                 batch_size: int=32,
                 device: Optional[str]=None,
                 **kwargs):
        super().__init__(policy=MlpPolicy, env=env, learning_rate=0.0, **kwargs)
        self.env = self.env.envs[0] if hasattr(self.env, 'envs') else self.env
        self.n_sims = n_sims
        self.c = c
        self.beta = beta
        self.gamma = gamma
        self.policy = None  # Not used in planning.
        self.action_space = self.env.action_space
        # device and prior net initialization
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device is None else th.device(device)
        policy_net_kwargs['device'] = str(self.device)
        self.prior_net = EpistemicRiskSeekerPolicy(**policy_net_kwargs)
        # replay buffer
        self.replay_buffer = deque(maxlen=replay_size)
        self.batch_size = batch_size

    def _setup_model(self) -> None:
        # no standard policy/value networks to set up
        pass
    def simulate(self, state: Any, action: int ,env) -> Tuple[Any, Any, float, bool, dict]:
        temp = copy.deepcopy(env)
        restore_env_state(temp, state)
        obs, r, done, trunc, info = temp.step(action)
        return clone_env_state(temp), obs, r, done or trunc, info

    def plan(self, obs: Any) -> int:
        if hasattr(self, "eval_env"):
            self.eval_env = self.eval_env.envs[0] if hasattr(self.eval_env, 'envs') else self.eval_env
            env = self.eval_env
        else:
            env = self.env
        state = clone_env_state(env)
        if hasattr(env, "get_obs"):
            obs = env.get_obs()
        else:
            obs = env.reset()[0]
        # Dynamically compute action mask from the current environment state.
        mask = get_action_masks(env)
        root = EpistemicMCTSNode(state, obs, mask=mask, prior=0.0)
        for _ in range(self.n_sims):
            node = root
            # selection
            while node.children and \
                  not node.done and \
                  node.is_fully_expanded(self):
                node = node.best_child(self, self.c)
            # expansion
            if not node.done:
                node = node.expand(self,env=env)
            # evaluation
            v = self.prior_net.get_value([node.obs])[0]
            u = self.prior_net.get_uncertainty([node.obs])[0]
            node.backup(v, u)
            self.replay_buffer.append((node.obs, v, u))
        # choose action by highest visit count
        best_action = None
        best_visits = -1
        for a, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = a
        if best_action is None:
            legal = get_legal_actions(obs, mask, self.action_space)
            best_action = random.choice(legal) if legal else 0
        return best_action

    def _evaluate(self, node: EpistemicMCTSNode) -> Tuple[float, float]:
        # use prior_net to get value & uncertainty
        v = self.prior_net.get_value([node.obs])[0]
        u = self.prior_net.get_uncertainty([node.obs])[0]
        return v, u

    def learn(self,
              total_timesteps: int,
              callback: Optional[object] = None,
              log_interval: int = 1,
              reset_num_timesteps: bool = True,
              progress_bar: bool = False) -> "EpistemicMCTS":
        # mirror Stable Baselines callback structure
        # self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_timesteps = self.num_timesteps  # resume if needed
        iteration = 0
        # Create or reuse a persistent SummaryWriter
        if not hasattr(self, 'writer') and callback is not None:
            self.writer = callback.writer #SummaryWriter(log_dir=log_dir)
        if callback is not None:
            self._logger = callback.writer
            callback = self._init_callback(callback, progress_bar)
            callback.log_interval = log_interval
            callback.on_training_start(locals(), globals())
        self.start_time = time.time_ns()
        self._num_timesteps_at_start = self.num_timesteps

        obs = self.env.reset()[0]
        episode_reward = 0.0
        while self.num_timesteps < total_timesteps:
            action = self.plan(obs)
            obs, reward, done, trunc, info = self.env.step(action)
            episode_reward += reward
            self.replay_buffer.append((obs, action, reward, done))
            self.num_timesteps += 1

            if callback is not None and callback._on_step() is False:
                callback.on_training_end()
                return self

            if done or trunc:
                if callback is not None:
                    callback.on_rollout_end()
                self._update_prior_net()
                obs = self.env.reset()[0]
                episode_reward = 0.0

            if iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.writer.add_scalar("time/iterations", iteration, self.num_timesteps)
                self.writer.add_scalar("time/fps", fps, self.num_timesteps)
                self.writer.add_scalar("time/total_timesteps", self.num_timesteps, self.num_timesteps)
                self.writer.flush()
            iteration += 1
        if callback is not None:
            callback.on_training_end()
        return self

    def _update_prior_net(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # sample: replay_buffer stores (obs, v, u)
        batch = random.sample(self.replay_buffer, self.batch_size)
        loss = compute_mcts_losses(batch, self.prior_net, self.gamma)
        # gradient step
        self.prior_net.optimizer.zero_grad()
        loss.backward()
        self.prior_net.optimizer.step()

    def predict(self, obs: Any, state=None, episode_start=None, deterministic=False,use_ensemble=False) -> Tuple[np.ndarray, None]:
        a = self.plan(obs)
        return np.array([a]), None
    
    def save(self, path: Union[str, os.PathLike]) -> None:
        if not str(path).endswith(".pt"):
            path = f"{path}.pt"
        checkpoint = {
            "n_sims":        self.n_sims,
            "c":             self.c,
            "beta":          self.beta,
            "gamma":         self.gamma,
            "batch_size":    self.batch_size,
            "num_timesteps": self.num_timesteps,
            "prior_state":   self.prior_net.state_dict(),
            "optim_state":   self.prior_net.optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer),
        }
        th.save(checkpoint, path)
        print(f"Saved EpistemicMCTS checkpoint to {path}")

    @classmethod
    def load(cls,
             path: Union[str, os.PathLike],
             env: gym.Env,
             policy_net_kwargs: Dict[str, Any],
             **kwargs) -> "EpistemicMCTS":
        with th.serialization.safe_globals(["numpy.core.multiarray.scalar"]):
            ckpt = th.load(path, weights_only=False)
        # Recreate the agent
        agent = cls(env,
                    policy_net_kwargs,
                    n_sims=ckpt["n_sims"],
                    c=ckpt["c"],
                    beta=ckpt["beta"],
                    gamma=ckpt["gamma"],
                    batch_size=ckpt["batch_size"],
                    **kwargs)
        agent.num_timesteps = ckpt.get("num_timesteps", 0)
        agent.prior_net.load_state_dict(ckpt["prior_state"])
        agent.prior_net.optimizer.load_state_dict(ckpt["optim_state"])
        agent.replay_buffer = deque(ckpt["replay_buffer"], maxlen=agent.replay_buffer.maxlen)
        print(f"Loaded EpistemicMCTS checkpoint from {path}")
        return agent
