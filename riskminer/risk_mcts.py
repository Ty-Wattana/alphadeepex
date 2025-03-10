import math
import random
import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

# Import BaseAlgorithm and a dummy policy class.
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from sb3_contrib.common.maskable.utils import get_action_masks

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
        return [i for i, valid in enumerate(action_mask) if valid]
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
                 n_simulations: int = 100,
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
              callback: Optional[Any] = None,
              tb_log_name: str = "run",
              reset_num_timesteps: bool = True,
              progress_bar: bool = False) -> "MCTSAlgorithm":
        self.num_timesteps = 0
        if callback is not None:
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
