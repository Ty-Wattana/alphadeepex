import json
import os
import warnings
import gym
from typing import Optional, Tuple, List, Any, Union, TypeVar
from datetime import datetime
from pathlib import Path
from openai import OpenAI

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

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
from stable_baselines3 import HerReplayBuffer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import numpy as np

from boot_alpha.boot_dqn_sb3 import BootstrappedDQN
from boot_alpha.reward_dense_env import AlphaEnvDense

SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")


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
        
        if self.num_timesteps % 100 == 0:
            self.save_checkpoint()

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
        return self.training_env.envs[0].unwrapped  # type: ignore

class ModifiedEvalCallback(EventCallback):
    """
    EvalCallback variant that:
      - Uses only a single validation env
      - Logs validation, training pool stats, and test-set metrics
      - Saves only the best validation model
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        test_calculators: List[Any],  # QLibStockDataCalculator
        n_eval_episodes: int = 100,
        eval_freq: int = 100,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback=None, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.best_ic_mean = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.test_calculators = test_calculators

        # Wrap single env to VecEnv if needed
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path

    def _init_callback(self) -> None:
        # Create directory for the best model
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        # Warn if training and eval types differ
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                f"Training and eval env are not same type: {type(self.training_env)} vs {type(self.eval_env)}"
            )

    def save_checkpoint(self):
        path = os.path.join(self.best_model_save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.env_core.pool.to_json_dict(), f)
    def _on_step(self) -> bool:
        # Only run evaluation every eval_freq calls
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync normalization if needed
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except Exception:
                    pass

            # --- Log training pool stats ---
            pool = self.env_core.pool
            size = pool.size
            sig_count = (np.abs(pool.weights[:size]) > 1e-4).sum()
            best_ic_ret = pool.best_ic_ret
            eval_cnt = pool.eval_cnt
            self.logger.record("train/pool_size", size)
            self.logger.record("train/pool_significant", int(sig_count))
            self.logger.record("train/best_ic_ret", float(best_ic_ret))
            self.logger.record("train/eval_cnt", int(eval_cnt))

            # --- Evaluate on validation env ---
            self.model.mask_env = self.eval_env
            ep_rews, ep_lens = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn
            )
            del self.model.mask_env
            mean_reward = float(np.mean(ep_rews))
            mean_length = float(np.mean(ep_lens))
            self.logger.record("validation/mean_reward", mean_reward)
            self.logger.record("validation/mean_ep_length", mean_length)
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # --- Evaluate on test calculators ---
            n_days = sum(cal.data.n_days for cal in self.test_calculators)
            ic_mean = 0.0
            rank_ic_mean = 0.0
            for i, cal in enumerate(self.test_calculators, start=1):
                ic, rank_ic = pool.test_ensemble(cal)
                self.logger.record(f"test/ic_{i}", float(ic))
                self.logger.record(f"test/rank_ic_{i}", float(rank_ic))
                weight = cal.data.n_days / n_days
                ic_mean += ic * weight
                rank_ic_mean += rank_ic * weight
            self.logger.record("test/ic_mean", float(ic_mean))
            self.logger.record("test/rank_ic_mean", float(rank_ic_mean))

            # Save model on new best validation reward
            if mean_reward > self.best_mean_reward:
                # if self.verbose > 0:
                #     print(f"New best validation reward: {mean_reward:.2f} -> saving model.")
                # if self.best_model_save_path is not None:
                #     path = os.path.join(self.best_model_save_path, "best_model")
                #     self.model.save(path)  # type: ignore

                self.save_checkpoint()
                self.best_mean_reward = mean_reward

            if ic_mean > self.best_ic_mean:

                self.save_checkpoint()
                self.best_ic_mean = ic_mean

        return True

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        # No child callbacks to update
        pass

    @property
    def env_core(self):
        # Access to underlying env_core for pool
        return self.training_env.envs[0].unwrapped  # type: ignore



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
        "boot" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/boot_dqn", name_prefix)
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
        constrain = True,
        penalty = True
        # her = True
    )
    # checkpoint_callback = CustomCallback(
    #     save_path=save_path,
    #     test_calculators=calculators[1:],
    #     verbose=1,
    #     chat_session=inter,
    #     llm_every_n_steps=llm_every_n_steps,
    #     drop_rl_n=drop_rl_n
    # )

    eval_pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[1],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )

    eval_env = AlphaEnvDense(
        pool=eval_pool,
        device=device,
        print_expr=True,
        # constrain = True,
        # penalty = True,
        # her = True
    )

    checkpoint_callback = ModifiedEvalCallback(
        best_model_save_path=save_path,
        test_calculators=calculators[1:],
        eval_env=eval_env,
    )
    model = BootstrappedDQN(
        policy="MlpPolicy",
        # policy="MultiInputPolicy",
        env=env,
        train_freq=(1, "episode"),
        num_bootstrapped_nets=20,
        mask_prob=0.8,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.1,
        verbose=1,
        learning_starts=1000,
        # replay_buffer_class=HerReplayBuffer,
        # replay_buffer_kwargs=dict(
        #     n_sampled_goal=4,
        #     goal_selection_strategy="future",
        #     ),
        tensorboard_log="./out/boot_tensorboard",
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
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