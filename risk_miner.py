import json
import os
from typing import Optional, Tuple, List, Any
from datetime import datetime
from pathlib import Path
from openai import OpenAI

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction
from alphagen.rl.policy import LSTMSharedNet
from riskminer.risk_mcts import MCTSAlgorithm, RiskMCTSAlgorithm
from sb3_contrib import QRDQN
from sb3_contrib.common.maskable.utils import get_action_masks
from typing import Dict

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import numpy as np
from typing import List, Optional
from typing import TypeVar
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from gymnasium import spaces
from boot_alpha.reward_dense_env import AlphaEnvDense
from stable_baselines3.common.noise import ActionNoise

SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

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

class MaskQRDQN(QRDQN):
    """
    A maskable version of QRDQN that overrides the predict method to incorporate an action mask.
    It assumes that the environment can provide an action mask via a helper function.
    """

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            action_masks = torch.as_tensor(get_action_masks(self.env)).cpu().numpy()
            valid_actions = [np.where(action_masks)[0]]
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([np.random.choice(valid_actions)])
            else:
                action = np.array([np.random.choice(valid_actions)])
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            action_masks = torch.as_tensor(get_action_masks(self.env)).cpu().numpy()
            valid_actions = [np.where(action_masks)[0]]
            unscaled_action = np.array([np.random.choice(valid_actions)])
            # unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
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
    # def predict(
    #     self,
    #     observation: Union[np.ndarray, Dict[str, np.ndarray]],
    #     state: Optional[Tuple[np.ndarray, ...]] = None,
    #     episode_start: Optional[np.ndarray] = None,
    #     deterministic: bool = False,
    # ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
    #     # If in exploration mode, sample random valid actions.
    #     if not deterministic and np.random.rand() < self.exploration_rate:
    #         if self.policy.is_vectorized_observation(observation):
    #             if isinstance(observation, dict):
    #                 n_batch = next(iter(observation.values())).shape[0]
    #             else:
    #                 n_batch = observation.shape[0]
    #             mask = get_action_masks(self.env)  # expected shape: (n_batch, num_actions)
    #             if mask.ndim == 1:
    #                 mask = np.tile(mask, (n_batch, 1))
    #             actions = []
    #             for i in range(n_batch):
    #                 valid_actions = np.nonzero(mask[i])[0]
    #                 if len(valid_actions) == 0:
    #                     valid_actions = np.arange(self.action_space.n)
    #                 actions.append(np.random.choice(valid_actions))
    #             action = np.array(actions)
    #         else:
    #             mask = get_action_masks(self.env)  # expected shape: (num_actions,)
    #             valid_actions = np.nonzero(mask)[0]
    #             if len(valid_actions) == 0:
    #                 valid_actions = np.arange(self.action_space.n)
    #             action = np.random.choice(valid_actions)
    #         return action, state
    #     else:
    #         # Convert observation to tensor.
    #         if isinstance(observation, dict):
    #             obs_tensor = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
    #                         for k, v in observation.items()}
    #         else:
    #             obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
                
    #         # Call forward without extra arguments.
    #         quantiles = self.policy.forward(obs_tensor, deterministic=deterministic)
    #         new_state = None  # non-recurrent policy; no hidden state is returned
    #         # Compute mean Q-values over quantiles.
    #         quantiles = quantiles.float()
    #         q_values = quantiles.mean(dim=-1).cpu().numpy()  # shape: (batch, num_actions) or (num_actions,)
    #         # Obtain the action mask.
    #         mask = get_action_masks(self.env)
    #         if q_values.ndim > 1 and mask.ndim == 1:
    #             mask = np.tile(mask, (q_values.shape[0], 1))
    #         # Set invalid actions' Q-values to -infinity.
    #         q_values_masked = np.where(mask.astype(bool), q_values, -np.inf)
    #         # Select the action with the highest valid Q-value.
    #         if q_values_masked.ndim > 1:
    #             action = np.argmax(q_values_masked, axis=1)
    #         else:
    #             action = np.argmax(q_values_masked)
    #         return action, new_state
        
    # def _sample_action(
    #     self,
    #     learning_starts: int,
    #     action_noise: Optional[any] = None,
    #     n_envs: int = 1,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Sample an action according to the exploration policy,
    #     ensuring that only valid actions are considered.
    #     """
    #     if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
    #         # Warmup phase: sample random valid actions.
    #         if isinstance(self.action_space, spaces.Discrete):
    #             mask = get_action_masks(self.env)  # expected shape: (num_actions,) or (n_envs, num_actions)
    #             if mask.ndim == 1:
    #                 mask = np.tile(mask, (n_envs, 1))
    #             actions = []
    #             for i in range(n_envs):
    #                 valid_actions = np.nonzero(mask[i])[0]
    #                 if len(valid_actions) == 0:
    #                     valid_actions = np.arange(self.action_space.n)
    #                 actions.append(np.random.choice(valid_actions))
    #             unscaled_action = np.array(actions)
    #         else:
    #             unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
    #     else:
    #         unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        
    #     if isinstance(self.action_space, spaces.Box):
    #         scaled_action = self.policy.scale_action(unscaled_action)
    #         if action_noise is not None:
    #             scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
    #         buffer_action = scaled_action
    #         action = self.policy.unscale_action(scaled_action)
    #     else:
    #         # For discrete action spaces, no need to scale.
    #         buffer_action = unscaled_action
    #         action = buffer_action
    #     return action, buffer_action


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out/llm-tests/interaction"
    parser = build_parser()
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url="https://api.ai.cs.ac.cn/v1"),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )


class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.chat_session is not None:
            self._try_use_llm()

        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        self.logger.record('pool/expr_len', len(self.env_core._tokens))
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.logger.record(f'test/rank_ic_mean', rank_ic_test_mean)

        self.save_checkpoint()
        # if self.num_timesteps % 2048 == 0:
        #     self.save_checkpoint()

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Save the local and global context for later use
        self.locals = locals_
        self.globals = globals_
        # For MCTS, there is no model, so we initialize our timestep counter directly.
        # You can either set it to zero or extract it from the algorithm if it stores a counter.
        self.num_timesteps = locals_.get("self", None).num_timesteps if "self" in locals_ and hasattr(locals_["self"], "num_timesteps") else 0
        self._on_training_start()
    
    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_json_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1
        
        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}, obj={self.pool.best_ic_ret:.4f}")

        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    @property
    def pool(self) -> LinearAlphaPool:
        assert(isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        if hasattr(self.training_env, "envs"):
            return self.training_env.envs[0].unwrapped
        else:
            return self.training_env.unwrapped


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3
):
    reseed_everything(seed)
    initialize_qlib("~/.qlib/qlib_data/cn_data")

    llm_replace_n = 0 if not use_llm else llm_replace_n
    print(f"""[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}""")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # tag = "rlv2" if llm_add_subexpr == 0 else f"afs{llm_add_subexpr}aar1-5"
    tag = (
        "agpt" if alphagpt_init else
        "mcts" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/risk_miner", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )

    segments = [
        ("2012-01-01", "2021-12-31"),
        ("2022-01-01", "2022-06-30"),
        ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30")
    ]
    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    chat, inter, pool = None, None, build_pool([])
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            calculator_train=calculators[0], calculators_test=calculators[1:],
            replace_k=llm_replace_n, forgetful=True
        )
        pool = inter.run()

    env = AlphaEnvDense(
        pool=pool,
        device=device,
        print_expr=True,
        # penalty = True,
        # constrain = True,
        # her = True
    )
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )
    # policy_kwargs = dict(n_quantiles=50)
    # model = MCTSAlgorithm(
    #     env=env, 
    #     policy_kwargs=policy_kwargs, 
    #     verbose=1,
    #     tensorboard_log="./out/riskminer_tensorboard",
    #     )
    
    # model.learn(
    #             total_timesteps=steps, 
    #             callback=checkpoint_callback, 
    #             tb_log_name=name_prefix
    #             )
    
    policy_net_kwargs = {
        "input_dim": env.observation_space.shape[0],  # dimension of observation
        "hidden_dim": 64,                              # GRU hidden dimension
        "num_layers": 1,                               # number of GRU layers
        "action_dim": env.action_space.n,              # number of actions
        "mlp_hidden_sizes": [32, 32],                  # MLP hidden layers
        "lr": 0.001                                    # learning rate
    }
    model = RiskMCTSAlgorithm(
        env=env,
        policy_net_kwargs=policy_net_kwargs,
        n_simulations=10,   # number of MCTS simulations per planning step
        c_param=1.41,
        alpha=0.7,          # risk-seeking quantile level
        replay_size=1000,   # smaller replay size for demo purposes
        batch_size=32,
        tensorboard_log="./out/riskminer_tensorboard",
        gamma=1.0
    )

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        log_interval=10,
        tb_log_name=name_prefix,
        reset_num_timesteps=True,
        progress_bar=True
    )


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000
):
    """
    :param random_seeds: Random seeds
    :param pool_capacity: Maximum size of the alpha pool
    :param instruments: Stock subset name
    :param alphagpt_init: Use an alpha set pre-generated by LLM as the initial pool
    :param use_llm: Enable LLM usage
    :param drop_rl_n: Drop n worst alphas before invoke the LLM
    :param steps: Total iteration steps
    :param llm_every_n_steps: Invoke LLM every n steps
    """
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds, )
    default_steps = {
        10: 200_000,
        20: 250_000,
        50: 300_000,
        100: 350_000
    }
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=default_steps[int(pool_capacity)] if steps is None else int(steps),
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps
        )


if __name__ == '__main__':
    main()