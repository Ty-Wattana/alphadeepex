import json
import os
from typing import Optional, Tuple, List
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import fire

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction
from stable_baselines3.common.type_aliases import MaybeCallback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import TensorDict
from typing import List, Optional, Iterable
from sb3_contrib.common.maskable.utils import get_action_masks
from typing import TypeVar
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise
from gymnasium import spaces

SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")

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

    def _polyak_update(
        online_params: Iterable[torch.Tensor],
        target_params: Iterable[torch.Tensor],
        tau: float,
    ) -> None:
        """
        Perform a Polyak averaging update on the target network parameters.
        
        Args:
            online_params: Parameters of the online network.
            target_params: Parameters of the target network.
            tau: The interpolation factor (e.g., 0.005 for soft updates).
        """
        for online_param, target_param in zip(online_params, target_params):
            target_param.data.mul_(1 - tau)  # target = (1 - tau) * target
            target_param.data.add_(tau * online_param.data)  # target += tau * online
    def _update_target_network(self) -> None:
        """
        Update the target networks for all bootstrapped Q-networks.
        """
        for i in range(self.num_bootstrapped_nets):
            self._polyak_update(
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
            q_values = torch.stack([q_net(observation_tensor) for q_net in self.q_networks])
            q_values = q_values.mean(dim=0)  # Average Q-values across all networks

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
        Override the _sample_action method to remove epsilon greedy action selection.
        """
        
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
        "rl" if not use_llm else
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

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=True
    )
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )
    model = BootstrappedDQN(
        policy="MlpPolicy",
        env=env,
        num_bootstrapped_nets=10,
        mask_prob=0.8,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./out/boot_tensorboard"
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
