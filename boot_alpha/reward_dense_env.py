from alphagen.data.tokens import *
from alphagen.config import MAX_EXPR_LENGTH
import math
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnvWrapper
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.utils import reseed_everything
from alphagen.data.tree import ExpressionBuilder
import gymnasium as gym
from gymnasium import spaces
# from gym import spaces
# import gym

from typing import Optional, Tuple, List, Dict

import numpy as np
import re
from alphagen.config import *

SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP

class AlphaDenseEnv(AlphaEnvCore):

    """"
    Override the step method to make a reward-dense environment.
    """
    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        len_penalty = (len(self._tokens)/MAX_EXPR_LENGTH)/2
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            reward -= len_penalty
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = self._evaluate() if self._builder.is_valid() else 0
            reward -= len_penalty
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.
            reward -= len_penalty

        if math.isnan(reward):
            reward = 0.

        return self._tokens, reward, done, False, self._valid_action_types()
    
class AlphaStrictEnvDense(AlphaEnvWrapper):

    """"
    Override action masks to be more strict.
    """

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0              # Operators
        for i in range(offset, offset + SIZE_OP):
            ops = OPERATORS[i - offset]
            if valid['op'][ops.category_type()]:
                res[i] = True
                if len(self.env._tokens) >= 2:
                    last_two_op = str(self.env._tokens[-2])
                    last_op = str(self.env._tokens[-1])
                    str_ops = str_ops = re.findall(r"\.([^']+)'>", str(ops))
                    str_ops = str_ops[0].split('.')[-1]
                    if str_ops == last_two_op or ops == last_op:
                       res[i] = False

        offset += SIZE_OP
        if valid['select'][1]:  # Features
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid['select'][2]:  # Constants
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid['select'][3]:  # Delta time
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid['select'][1]:  # Sub-expressions
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        if valid['select'][4]:  # SEP
            res[offset] = True
        return res    

class AlphaEnvGoal(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        pool: 'AlphaPoolBase',
        desired_goal: float,
        device: torch.device = torch.device('cuda:0'),
        print_expr: bool = False,
        tolerance: float = 1e-3
    ):
        super().__init__()
        self.pool = pool
        self.desired_goal = desired_goal  # target value for the expression
        self._print_expr = print_expr
        self._device = device
        self.tolerance = tolerance
        self.eval_cnt = 0
        self.MAX_EXPR_LENGTH = MAX_EXPR_LENGTH

        # Define the action space.
        num_actions = SIZE_ACTION 
        # self.action_space = spaces.Dict({'action': spaces.Discrete(num_actions)})

        # Define the observation space as a Dict containing:
        # - "observation": a fixed-length array of tokens (padded with 0)
        # - "achieved_goal": the evaluated result (float wrapped in an array of shape (1,))
        # - "desired_goal": the target value (float wrapped in an array of shape (1,))
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=num_actions - 1, shape=(self.MAX_EXPR_LENGTH,), dtype=np.int32),
            'achieved_goal': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32)
        })

        # Define the action space (here assuming a dict action space)
        self.action_space = spaces.Dict({'action': spaces.Discrete(num_actions)})

        self.reset()

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Tuple[Dict, dict]:
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        self.eval_cnt = 0
        self._achieved_goal = 0.0
        obs = self._get_obs()
        info = self._valid_action_types()
        if return_info:
            return obs, info
        return obs
    
    def _token_to_action(self, token: Token) -> int:
        # OperatorToken: action is the index in the OPERATORS list.
        if isinstance(token, OperatorToken):
            try:
                op_index = OPERATORS.index(token.operator)  # adjust attribute name as needed
            except ValueError:
                raise ValueError("Operator not found in OPERATORS.")
            return op_index

        # FeatureToken: action is SIZE_OP offset plus the int value of the FeatureType.
        elif isinstance(token, FeatureToken):
            # Assuming token.feature is an instance of FeatureType convertible to int.
            return SIZE_OP + int(token.feature)

        # ConstantToken: action is SIZE_OP + SIZE_FEATURE offset plus the index in CONSTANTS.
        elif isinstance(token, ConstantToken):
            try:
                const_index = CONSTANTS.index(token.constant)  # adjust attribute name as needed
            except ValueError:
                raise ValueError("Constant not found in CONSTANTS.")
            return SIZE_OP + SIZE_FEATURE + const_index

        # DeltaTimeToken: action is SIZE_OP + SIZE_FEATURE + SIZE_CONSTANT offset plus the index in DELTA_TIMES.
        elif isinstance(token, DeltaTimeToken):
            try:
                dt_index = DELTA_TIMES.index(token.delta_time)  # adjust attribute name as needed
            except ValueError:
                raise ValueError("Delta time not found in DELTA_TIMES.")
            return SIZE_OP + SIZE_FEATURE + SIZE_CONSTANT + dt_index

        # ExpressionToken: action is SIZE_OP + SIZE_FEATURE + SIZE_CONSTANT + SIZE_DELTA_TIME offset plus the index in self.subexprs.
        elif isinstance(token, ExpressionToken):
            try:
                expr_index = self.subexprs.index(token.expr)  # adjust attribute name as needed
            except ValueError:
                raise ValueError("Expression not found in subexpressions.")
            return SIZE_OP + SIZE_FEATURE + SIZE_CONSTANT + SIZE_DELTA_TIME + expr_index

        # SequenceIndicatorToken: for now, only supporting the SEP indicator.
        elif isinstance(token, SequenceIndicatorToken):
            if token.sequence_indicator != SequenceIndicatorType.SEP:
                raise ValueError("Unsupported SequenceIndicatorToken type.")
            return SIZE_OP + SIZE_FEATURE + SIZE_CONSTANT + SIZE_DELTA_TIME + len(self.subexprs)

        else:
            raise ValueError("Unknown token type encountered.")

    def _get_obs(self) -> Dict:
        # Pad the tokens to a fixed length (if needed).
        tokens_without_beg = self._tokens[1:]
        actions_without_beg = [self._token_to_action(token) for token in tokens_without_beg]
        padded_tokens = actions_without_beg + [0] * (self.MAX_EXPR_LENGTH - len(self._tokens))
        return {
            'observation': np.array(padded_tokens, dtype=np.int32),
            'achieved_goal': np.array([self._achieved_goal], dtype=np.float32),
            'desired_goal': np.array([self.desired_goal], dtype=np.float32)
        }

    def step(self, action: 'Token') -> Tuple[Dict, float, bool, bool, dict]:
        done = False
        reward = 0.0
        len_penalty = (len(self._tokens)/MAX_EXPR_LENGTH)/2

        # If a special token indicating termination is received, evaluate the expression.
        if (isinstance(action, SequenceIndicatorToken) and 
                action.indicator == SequenceIndicatorType.SEP):
            self._achieved_goal = self._evaluate()
            self._achieved_goal -= len_penalty
            done = True
            reward = self.compute_reward(self._achieved_goal, self.desired_goal, self._valid_action_types())
        elif len(self._tokens) < self.MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            if self._builder.is_valid():
                self._achieved_goal = self._evaluate()
                self._achieved_goal -= len_penalty
                # reward = self.compute_reward(self._achieved_goal, self.desired_goal, self._valid_action_types())
                reward = 0.0
            else:
                reward = 0.0
                self._achieved_goal = 0.0
            
        else:
            done = True
            if self._builder.is_valid():
                self._achieved_goal = self._evaluate()
                self._achieved_goal -= len_penalty
                reward = self.compute_reward(self._achieved_goal, self.desired_goal, self._valid_action_types())
            else:
                reward = -1.0
                self._achieved_goal = 0.0

        obs = self._get_obs()
        info = self._valid_action_types()
        # Following gymnasium's API: return (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Ensure inputs are tensors (or work with numpy arrays, but here we use torch)
        if not isinstance(achieved_goal, torch.Tensor):
            achieved_goal = torch.tensor(achieved_goal, dtype=torch.float32, device=self._device)
        if not isinstance(desired_goal, torch.Tensor):
            desired_goal = torch.tensor(desired_goal, dtype=torch.float32, device=self._device)
            
        # Compute the absolute difference
        diff = torch.abs(achieved_goal - desired_goal)
        
        # Return a reward of 0 if within tolerance, else -1
        # This uses torch.where to be fully vectorized.
        reward = torch.where(diff < self.tolerance,
                            torch.tensor(0.0, device=diff.device),
                            torch.tensor(-1.0, device=diff.device))
        
        return reward.cpu().numpy() # Convert to NumPy array before returning

    def _evaluate(self) -> float:
        expr = self._builder.get_tree()
        if self._print_expr:
            print(expr)
        try:
            ret = self.pool.try_new_expr(expr)
            self.eval_cnt += 1
            return ret
        except OutOfDataRangeError:
            return 0.0

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)
        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_featured_expr()
        valid_stop = self._builder.is_valid()
        return {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
    
    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode='human'):
        # Implement rendering if needed.
        pass

class AlphaEnvHERWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: spaces.Discrete
    observation_space: spaces.Dict
    counter: int

    def __init__(self, env: AlphaEnvCore, subexprs: Optional[List[Expression]] = None):
        super().__init__(env)
        self.subexprs = subexprs or []
        self.size_action = SIZE_ACTION + len(self.subexprs)
        self.action_space = spaces.Discrete(self.size_action)
        # Instead of a Box, create a Dict observation space:
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=0, 
                high=self.size_action + SIZE_NULL - 1,
                shape=(MAX_EXPR_LENGTH,), 
                dtype=np.uint8
            ),
            'achieved_goal': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            ),
            'desired_goal': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            )
        })

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        self.counter = 0
        # Initialize state as a zero array for token indices
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        # Reset the underlying environment (which might update its internal desired_goal, etc.)
        self.env.reset(**kwargs)
        # Here, we assume the underlying env has attributes for desired/achieved goals;
        # if not, default to 0.
        achieved_goal = getattr(self.env, '_achieved_goal', 0.0)
        desired_goal = getattr(self.env, 'desired_goal', 0.0)
        obs = {
            'observation': self.state,
            'achieved_goal': np.array([achieved_goal], dtype=np.float32),
            'desired_goal': np.array([desired_goal], dtype=np.float32)
        }
        return obs, {}

    def step(self, action: int):
        token = self.action(action)
        # Step the underlying env with the token action
        _, reward, done, truncated, info = self.env.step(token)
        # Update the state with the chosen action
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        # You can adjust the reward as needed
        new_reward = self.reward(reward)
        # Obtain achieved and desired goals from the underlying env (if available)
        achieved_goal = getattr(self.env, '_achieved_goal', 0.0)
        desired_goal = getattr(self.env, 'desired_goal', 0.0)
        obs = {
            'observation': self.state,
            'achieved_goal': np.array([achieved_goal], dtype=np.float32),
            'desired_goal': np.array([desired_goal], dtype=np.float32)
        }
        return obs, new_reward, done, truncated, info

    def action(self, action: int) -> Token:
        return self.action_to_token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()
        offset = 0  # Operators
        for i in range(offset, offset + SIZE_OP):
            if valid['op'][OPERATORS[i - offset].category_type()]:
                res[i] = True
        offset += SIZE_OP
        if valid['select'][1]:  # Features
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid['select'][2]:  # Constants
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid['select'][3]:  # Delta time
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid['select'][1]:  # Sub-expressions
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        if valid['select'][4]:  # SEP
            res[offset] = True
        return res

    def action_to_token(self, action: int) -> Token:
        if action < 0:
            raise ValueError("Negative action value encountered.")
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP
        if action < SIZE_FEATURE:
            return FeatureToken(FeatureType(action))
        action -= SIZE_FEATURE
        if action < SIZE_CONSTANT:
            return ConstantToken(CONSTANTS[action])
        action -= SIZE_CONSTANT
        if action < SIZE_DELTA_TIME:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= SIZE_DELTA_TIME
        if action < len(self.subexprs):
            return ExpressionToken(self.subexprs[action])
        action -= len(self.subexprs)
        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        assert False, "Invalid action index"

def AlphaEnvDense(pool: AlphaPoolBase, subexprs: Optional[List[Expression]] = None, constrain: bool = False, her=False, **kwargs):
    if constrain:
        if her:
            return AlphaEnvHERWrapper(AlphaEnvGoal(pool=pool,desired_goal=0.1, **kwargs), subexprs=subexprs)
        else:
            return AlphaStrictEnvDense(AlphaDenseEnv(pool=pool, **kwargs), subexprs=subexprs)
    else:
        if  her:
            return AlphaEnvHERWrapper(AlphaEnvGoal(pool=pool,desired_goal=0.1, **kwargs), subexprs=subexprs)
        else:
            return AlphaEnvWrapper(AlphaDenseEnv(pool=pool, **kwargs), subexprs=subexprs)