import math
import random
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

# Import BaseAlgorithm and a dummy policy class.
from stable_baselines3.common import utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from collections import deque

from alphagen.config import MAX_EXPR_LENGTH

# --- Helper: dynamic valid actions ---

def clone_env_state(env: Any) -> Any:
    # Here we assume env.get_state() returns a deep-copyable object
    return env.get_state()

def restore_env_state(env: Any, state: Any) -> None:
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

# --- MCTS Node ---
class MCTSNode:
    def __init__(self, env_state: Any, obs: Any, parent: Optional["MCTSNode"] = None, action: Optional[int] = None, action_mask: Optional[List[bool]] = None):
        self.env_state = env_state   # cloned environment state
        self.obs = obs               # observation (could be a list)
        self.parent = parent         # parent node
        self.action = action         # action taken from parent to reach here (None at root)
        self.children: Dict[int, "MCTSNode"] = {}  # mapping: action -> child node
        self.visits = 0              # number of visits
        self.value = 0.0             # cumulative reward
        self.done = False            # terminal flag
        # The valid actions at this node (binary mask)
        self.action_mask = action_mask
        # List of actions that have not yet been expanded
        self.untried_actions: List[int] = get_legal_actions(self.obs, self.action_mask, self._dummy_action_space())  # temporarily set; will be updated dynamically

    def _dummy_action_space(self) -> spaces.Space:
        # If the observation is a dict and contains "action_mask", we infer the length
        if isinstance(self.obs, dict) and "action_mask" in self.obs:
            return spaces.Discrete(len(self.obs["action_mask"]))
        # Otherwise, assume full mask (will be fixed later by the agent)
        return spaces.Discrete(10)  # default fallback; adjust as needed

    def is_fully_expanded(self, agent: "MCTSAlgorithm") -> bool:
        legal_actions = get_legal_actions(self.obs, self.action_mask, agent.action_space)
        return len(self.children) >= len(legal_actions)

    def expand(self, agent: "MCTSAlgorithm") -> "MCTSNode":
        # Refresh the untried actions based on the current observation.
        self.untried_actions = get_legal_actions(self.obs, self.action_mask, agent.action_space)
        # Remove actions that are already expanded.
        self.untried_actions = [a for a in self.untried_actions if a not in self.children]
        if not self.untried_actions:
            return self  # no legal action to expand
        idx = random.randrange(len(self.untried_actions))
        action = self.untried_actions.pop(idx)
        new_state, new_obs, reward, done, info = agent.simulate_action(self.env_state, action)
        # Dynamically compute child action mask by restoring state in a temporary env.
        temp_env = copy.deepcopy(agent.env)
        restore_env_state(temp_env, new_state)
        child_mask = get_action_masks(temp_env)
        child = MCTSNode(new_state, new_obs, parent=self, action=action, action_mask=child_mask)
        child.done = done
        self.children[action] = child
        return child

    def best_child(self, agent: "MCTSAlgorithm", c_param: float) -> "MCTSNode":
        best_score = -float("inf")
        best = None
        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                explore = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def rollout(self, agent: "MCTSAlgorithm", max_depth: int = MAX_EXPR_LENGTH) -> float:
        current_state = self.env_state
        current_obs = self.obs
        total_reward = 0.0
        depth = 0
        discount = agent.discount
        while depth < max_depth:
            # Dynamically compute action mask for current state.
            temp_env = copy.deepcopy(agent.env)
            restore_env_state(temp_env, current_state)
            mask = get_action_masks(temp_env)
            legal = get_legal_actions(current_obs, mask, agent.action_space)
            if not legal:
                break
            action = random.choice(legal)
            new_state, new_obs, reward, done, info = agent.simulate_action(current_state, action)
            total_reward += (discount ** depth) * reward
            depth += 1
            if done:
                break
            current_state = new_state
            current_obs = new_obs
        return total_reward

# --- MCTSAlgorithm Class ---
class MCTSAlgorithm(BaseAlgorithm):
    """
    A simple MCTS-based planner that mimics the interface of SB3's OffPolicyAlgorithm.
    This version supports dynamic action masking.
    
    """
    def __init__(self,
                 env: Union[str, gym.Env],
                 n_simulations: int = 10,
                 c_param: float = 1.41,
                 discount: float = 1.0,
                 **kwargs):
        # Pass a dummy policy (MlpPolicy) to satisfy BaseAlgorithm. (We won't use it.)
        super().__init__(policy=MlpPolicy, env=env, learning_rate=0.0, **kwargs)
        self.env = self.env.envs[0] if hasattr(self.env, "envs") else self.env
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.discount = discount
        self.num_timesteps = 0
        self.policy = None  # MCTS does not train a policy

    def _setup_model(self) -> None:
        # No model to set up since MCTS is a planning algorithm.
        pass

    def simulate_action(self, state: Any, action: int) -> Tuple[Any, Any, float, bool, dict]:
        temp_env = copy.deepcopy(self.env)
        restore_env_state(temp_env, state)
        obs, reward, done, truncated, info = temp_env.step(action)
        new_state = clone_env_state(temp_env)
        return new_state, obs, reward, done or truncated, info

    def _plan_action(self) -> int:
        state = clone_env_state(self.env)
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs()
        else:
            obs = self.env.reset()[0]
        # Dynamically compute action mask from the current environment state.
        mask = get_action_masks(self.env)
        # If obs is a dict, inject the mask; if not, we pass it separately.
        if isinstance(obs, dict):
            obs["action_mask"] = mask
        root = MCTSNode(state, obs, action_mask=mask)
        for _ in range(self.n_simulations):
            node = root
            while node.children and not node.done and node.is_fully_expanded(self):
                node = node.best_child(self, self.c_param)
            if not node.done:
                node = node.expand(self)
            reward = node.rollout(self)
            node.backpropagate(reward)
        best_action = None
        best_visits = -1
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        if best_action is None:
            legal = get_legal_actions(obs, mask, self.env.action_space)
            best_action = random.choice(legal) if legal else 0
        return best_action

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        action = self._plan_action()
        return np.array([action]), None

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              tb_log_name: str = "run",
              reset_num_timesteps: bool = True,
              progress_bar: bool = False) -> "MCTSAlgorithm":
        self.num_timesteps = 0
        iteration = 0
        if callback is not None:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
            callback = self._init_callback(callback, progress_bar)
            callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            obs = self.env.reset()
            done = False
            while not done and self.num_timesteps < total_timesteps:
                action = self._plan_action()
                obs, reward, done, truncated, info = self.env.step(action)
                self.num_timesteps += 1
                if callback is not None and callback._on_step() is False:
                    callback.on_training_end()
                    return self
                
                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                    fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                    self.logger.record("time/iterations", iteration, exclude="tensorboard")
                    
                    self.logger.record("time/fps", fps)
                    self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                    self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                    self.logger.dump(step=self.num_timesteps)

            callback.update_locals(locals())
            if callback is not None:
                callback.on_rollout_end()
        if callback is not None:
            callback.on_training_end()
        return self

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # MCTS does not update model parameters.
        pass

    def save(self, path: Union[str, os.PathLike]) -> None:
        config = {
            "n_simulations": self.n_simulations,
            "c_param": self.c_param,
            "discount": self.discount,
            "num_timesteps": self.num_timesteps,
        }
        with open(path, "w") as f:
            json.dump(config, f)
        print(f"Saved MCTS configuration to {path}")

    @classmethod
    def load(cls, path: Union[str, os.PathLike], env: gym.Env) -> "MCTSAlgorithm":
        with open(path, "r") as f:
            config = json.load(f)
        instance = cls(env, config["n_simulations"], config["c_param"], config["discount"])
        instance.num_timesteps = config.get("num_timesteps", 0)
        return instance

# --- Risk-Seeking MCTS Node ---

class RiskMCTSNode:
    def __init__(self, env_state: Any, obs: Any, parent: Optional["RiskMCTSNode"] = None,
                 action: Optional[int] = None, action_mask: Optional[Union[List[bool], np.ndarray]] = None,
                 prior: float = 0.0):
        self.env_state = env_state
        self.obs = obs
        self.parent = parent
        self.action = action
        self.children: Dict[int, "RiskMCTSNode"] = {}
        self.visits = 0
        self.value = 0.0
        self.done = False
        self.action_mask = action_mask
        self.untried_actions: List[int] = get_legal_actions(self.obs, self.action_mask, self._dummy_action_space())
        self.prior = prior  # Prior probability from risk policy

    def _dummy_action_space(self) -> spaces.Space:
        if self.action_mask is not None:
            return spaces.Discrete(len(self.action_mask))
        if isinstance(self.obs, dict) and "action_mask" in self.obs:
            return spaces.Discrete(len(self.obs["action_mask"]))
        return spaces.Discrete(10)

    def is_fully_expanded(self, agent: "RiskMCTSAlgorithm") -> bool:
        legal_actions = get_legal_actions(self.obs, self.action_mask, agent.action_space)
        return len(self.children) >= len(legal_actions)

    def expand(self, agent: "RiskMCTSAlgorithm") -> "RiskMCTSNode":
        self.untried_actions = get_legal_actions(self.obs, self.action_mask, agent.action_space)
        self.untried_actions = [a for a in self.untried_actions if a not in self.children]
        if not self.untried_actions:
            return self
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        new_state, new_obs, reward, done, info = agent.simulate_action(self.env_state, action)
        temp_env = copy.deepcopy(agent.env)
        restore_env_state(temp_env, new_state)
        child_mask = temp_env.action_masks() if hasattr(temp_env, "action_masks") else None
        # Obtain prior probability from the risk policy network for the new observation.
        prior = agent.risk_policy.get_prior(new_obs)[action] if hasattr(agent, "risk_policy") else 0.0
        child = RiskMCTSNode(new_state, new_obs, parent=self, action=action, action_mask=child_mask, prior=prior)
        child.done = done
        self.children[action] = child
        return child

    def best_child(self, agent: "RiskMCTSAlgorithm", c_param: float) -> "RiskMCTSNode":
        best_score = -float("inf")
        best = None
        total_visits = sum(child.visits for child in self.children.values())
        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                # Risk-based PUCT: incorporating the risk prior into the selection.
                score = exploit + (child.prior * math.sqrt(total_visits)) / (1 + child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def rollout(self, agent: "RiskMCTSAlgorithm", max_depth: int = MAX_EXPR_LENGTH) -> Tuple[float, List[float]]:
        current_state = self.env_state
        current_obs = self.obs
        total_reward = 0.0
        depth = 0
        discount = agent.gamma
        log_probs = []
        while depth < max_depth:
            temp_env = copy.deepcopy(agent.env)
            restore_env_state(temp_env, current_state)
            mask = temp_env.action_masks() if hasattr(temp_env, "action_masks") else None
            legal = get_legal_actions(current_obs, mask, agent.action_space)
            if not legal:
                break
            # Use risk policy to get prior probabilities.
            prior_probs = agent.risk_policy.get_prior(current_obs)
            legal_probs = np.array([prior_probs[a] for a in legal])
            if legal_probs.sum() == 0:
                legal_probs = np.ones_like(legal_probs)
            legal_probs /= legal_probs.sum()
            action = np.random.choice(legal, p=legal_probs)
            # Compute log probability for the chosen action.
            log_prob = np.log(legal_probs[legal.index(action)])
            log_probs.append(log_prob)
            new_state, new_obs, reward, done, info = agent.simulate_action(current_state, action)
            total_reward += (discount ** depth) * reward
            depth += 1
            if done:
                break
            current_state = new_state
            current_obs = new_obs
        return total_reward, log_probs

# --- Risk-Seeking Policy Network with Policy Gradient Update ---

class RiskSeekerPolicy(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_layers: int, 
                 action_dim: int, 
                 mlp_hidden_sizes: List[int] = [32, 32], 
                 lr: float = 0.001,
                 device: Optional[str] = None):
        super(RiskSeekerPolicy, self).__init__()
        # Determine device: use GPU if available, otherwise CPU.
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device is None else th.device(device)
        
        # GRU layer expects input of shape (batch, seq_len, input_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True).to(self.device)
        
        mlp_input_dim = hidden_dim
        layers = []
        last_dim = mlp_input_dim
        for size in mlp_hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, action_dim))
        self.mlp = nn.Sequential(*layers).to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, h_n = self.gru(x)
        feature = h_n[-1]  # Use the last layer's hidden state.
        logits = self.mlp(feature)
        return logits

    def get_prior(self, observation: Any) -> List[float]:
        # If observation is a 1D array, reshape it to (1, 1, input_dim)
        if isinstance(observation, np.ndarray) and observation.ndim == 1:
            observation = observation.reshape(1, 1, -1)
        # Create tensor on the correct device.
        x = th.tensor(observation, dtype=th.float32, device=self.device)
        logits = self.forward(x)
        # Compute softmax to get a probability distribution over actions.
        prior = th.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        return prior.tolist()

    def update_policy(self, trajectories: List[Dict[str, Any]], quantile: float, beta: float) -> float:
        loss = th.tensor(0.0, dtype=th.float32, requires_grad=True, device=self.device)
        for traj in trajectories:
            R = traj["return"]
            indicator = 1.0 if R <= quantile else 0.0
            # Convert list of log_probs to tensor.
            log_probs = th.tensor(traj["log_probs"], dtype=th.float32, device=self.device)
            loss = loss + (-indicator * log_probs.sum())
        loss = loss / len(trajectories)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# --- Risk-Seeking MCTS Algorithm Integrating Policy Gradients ---

class RiskMCTSAlgorithm(BaseAlgorithm):
    def __init__(self, 
                 env: gym.Env,
                 policy_net_kwargs: Dict[str, Any],
                 n_simulations: int = 10,
                 c_param: float = 1.41,
                 alpha: float = 0.7,
                 replay_size: int = 10000,
                 batch_size: int = 256,
                 gamma: float = 1.0,
                 device: Optional[str] = None,
                 **kwargs):
        # Use a dummy policy to satisfy BaseAlgorithm.
        super().__init__(policy=MlpPolicy, env=env, learning_rate=0.0, **kwargs)
        self.env = self.env.envs[0] if hasattr(self.env, "envs") else self.env
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.gamma = gamma  # Discount factor
        self.num_timesteps = 0
        self.policy = None  # Not used in planning.
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device is None else th.device(device)
        # Ensure the policy network gets the device information.
        policy_net_kwargs["device"] = str(self.device)
        self.risk_policy = RiskSeekerPolicy(**policy_net_kwargs)
        self.replay_buffer = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.alpha = alpha  # Risk-seeking quantile level.
        self.current_quantile = 0.0  # Initialize quantile threshold.

    def _setup_model(self) -> None:
        pass

    def simulate_action(self, state: Any, action: int) -> Tuple[Any, Any, float, bool, dict]:
        temp_env = copy.deepcopy(self.env)
        restore_env_state(temp_env, state)
        obs, reward, done, truncated, info = temp_env.step(action)
        new_state = clone_env_state(temp_env)
        return new_state, obs, reward, done or truncated, info

    def _plan_action(self) -> int:
        state = clone_env_state(self.env)
        obs = self.env.get_obs() if hasattr(self.env, "get_obs") else self.env.reset()[0]
        mask = self.env.action_masks() if hasattr(self.env, "action_masks") else None
        if isinstance(obs, dict):
            obs["action_mask"] = mask
        root = RiskMCTSNode(state, obs, action_mask=mask, prior=0.0)
        # Run MCTS simulations and record trajectories.
        for _ in range(self.n_simulations):
            node = root
            while node.children and not node.done and node.is_fully_expanded(self):
                node = node.best_child(self, self.c_param)
            if not node.done:
                node = node.expand(self)
            rollout_reward, rollout_log_probs = node.rollout(self)
            node.backpropagate(rollout_reward)
            trajectory = {"return": rollout_reward, "log_probs": rollout_log_probs}
            self.replay_buffer.append(trajectory)
        best_action = None
        best_visits = -1
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        if best_action is None:
            legal = get_legal_actions(obs, mask, self.env.action_space)
            best_action = random.choice(legal) if legal else 0
        return best_action

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        action = self._plan_action()
        return np.array([action]), None

    def update_risk_policy(self, beta: float):
        if len(self.replay_buffer) < self.batch_size:
            return
        trajectories = random.sample(self.replay_buffer, self.batch_size)
        loss = self.risk_policy.update_policy(trajectories, self.current_quantile, beta)
        returns = [traj["return"] for traj in trajectories]
        indicator_mean = np.mean([1 if R <= self.current_quantile else 0 for R in returns])
        self.current_quantile += beta * (1 - self.alpha - indicator_mean)
        # Log the loss to tensorboard
        self.writer.add_scalar("train/loss", loss, self.num_timesteps)

    def learn(self,
          total_timesteps: int,
          callback: Optional[object] = None,
          log_interval: int = 1,
          reset_num_timesteps: bool = True,
          progress_bar: bool = False) -> "RiskMCTSAlgorithm":
        # Create or reuse a persistent SummaryWriter
        if not hasattr(self, 'writer') and callback is not None:
            # log_dir = f"./out/riskminer_tensorboard/{tb_log_name}"
            self.writer = callback.writer #SummaryWriter(log_dir=log_dir)
        
        self.num_timesteps = self.num_timesteps  # Will use existing value if resuming
        iteration = 0
        if callback is not None:
            # self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
            self._logger = callback.writer
            callback = self._init_callback(callback, progress_bar)
            callback.log_interval = log_interval
            callback.on_training_start(locals(), globals())

        self.start_time = time.time_ns()
        self._num_timesteps_at_start = self.num_timesteps

        while self.num_timesteps < total_timesteps:
            obs = self.env.reset()
            done = False
            episode_reward = 0.0  # accumulate reward per episode
            while not done and self.num_timesteps < total_timesteps:
                action = self._plan_action()
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                self.num_timesteps += 1

                if callback is not None and callback._on_step() is False:
                    callback.on_training_end()
                    return self

                if log_interval is not None and iteration % log_interval == 0:
                    time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                    fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                    self.writer.add_scalar("time/iterations", iteration, self.num_timesteps)
                    self.writer.add_scalar("time/fps", fps, self.num_timesteps)
                    self.writer.add_scalar("time/time_elapsed", int(time_elapsed), self.num_timesteps)
                    self.writer.add_scalar("time/total_timesteps", self.num_timesteps, self.num_timesteps)
                    self.writer.flush()
                iteration += 1

            # Log the cumulative reward for the episode.
            self.writer.add_scalar("train/episode_reward", episode_reward, self.num_timesteps)
            if callback is not None:
                callback.update_locals(locals())
                callback.on_rollout_end()
            self.update_risk_policy(beta=0.01)
        if callback is not None:
            callback.on_training_end()
        return self

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # RiskMCTSAlgorithm does not use typical gradient steps during planning.
        pass

    def save(self, path: Union[str, os.PathLike]) -> None:
        # Ensure the path ends with the desired file extension, e.g., ".pt"
        if not str(path).endswith(".pt"):
            path = f"{path}.pt"
        checkpoint = {
            "n_simulations": self.n_simulations,
            "c_param": self.c_param,
            "gamma": self.gamma,
            "num_timesteps": self.num_timesteps,
            "alpha": self.alpha,
            "current_quantile": self.current_quantile,
            "risk_policy_state": self.risk_policy.state_dict(),
            "risk_policy_optimizer_state": self.risk_policy.optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer),
        }
        th.save(checkpoint, path)
        print(f"Saved RiskMCTS checkpoint to {path}")

    @classmethod
    def load(cls, path: Union[str, os.PathLike], env: gym.Env, policy_net_kwargs: Dict[str, Any]) -> "RiskMCTSAlgorithm":
        # Allow the global "numpy.core.multiarray.scalar" while loading.
        with th.serialization.safe_globals(["numpy.core.multiarray.scalar"]):
            checkpoint = th.load(path, weights_only=False)
        instance = cls(env, policy_net_kwargs,
                    n_simulations=checkpoint["n_simulations"],
                    c_param=checkpoint["c_param"],
                    alpha=checkpoint["alpha"],
                    gamma=checkpoint["gamma"])
        instance.num_timesteps = checkpoint.get("num_timesteps", 0)
        instance.current_quantile = checkpoint.get("current_quantile", 0.0)
        instance.risk_policy.load_state_dict(checkpoint["risk_policy_state"])
        instance.risk_policy.optimizer.load_state_dict(checkpoint["risk_policy_optimizer_state"])
        instance.replay_buffer = deque(checkpoint["replay_buffer"], maxlen=10000)
        print(f"Loaded RiskMCTS checkpoint from {path}")
        return instance