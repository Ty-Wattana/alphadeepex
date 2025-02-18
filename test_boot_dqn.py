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

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.preprocessing import get_obs_shape
import torch.nn as nn

class BootstrappedDQNQNetwork(BasePolicy):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, num_heads=10):
        super(BootstrappedDQNQNetwork, self).__init__(observation_space, action_space, features_extractor=features_extractor)
        
        self.num_heads = num_heads
        self.fc_layers = nn.ModuleList()
        
        for _ in range(num_heads):
            layers = []
            last_dim = features_dim
            for layer_size in net_arch:
                layers.append(nn.Linear(last_dim, layer_size))
                layers.append(nn.ReLU())
                last_dim = layer_size
            layers.append(nn.Linear(last_dim, action_space.n))  # Output layer for Q-values
            self.fc_layers.append(nn.Sequential(*layers))
        
    def forward(self, obs, head_idx=None):
        features = self.extract_features(obs)
        if head_idx is None:
            return torch.stack([head(features) for head in self.fc_layers], dim=0)  # Shape: (num_heads, batch, actions)
        else:
            return self.fc_layers[head_idx](features)  # Shape: (batch, actions)

    def select_action(self, obs, exploration=False):
        """ Select action using a randomly chosen head for exploration. """
        head_idx = torch.randint(0, self.num_heads, (1,)).item()
        q_values = self.forward(obs, head_idx)
        if exploration:
            return torch.randint(0, q_values.shape[-1], (1,)).item()  # Random action
        return torch.argmax(q_values, dim=-1).item()

class BootstrappedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, num_heads=10):
        super().__init__(buffer_size, observation_space, action_space, device)
        self.num_heads = num_heads
        self.head_masks = torch.randint(0, 2, (buffer_size, num_heads), dtype=torch.bool, device=device)

    def add(self, obs, action, reward, next_obs, done):
        idx = self.pos  # Current position in buffer
        super().add(obs, action, reward, next_obs, done)
        self.head_masks[idx] = torch.randint(0, 2, (self.num_heads,), dtype=torch.bool, device=self.device)

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        batch["head_masks"] = self.head_masks[batch["indexes"]]
        return batch

class BootstrappedDQN(OffPolicyAlgorithm):
    def __init__(self, policy, env, learning_rate=1e-3, gamma=0.99, tau=0.005, num_heads=10, buffer_size=100000, batch_size=32, train_freq=1, tensorboard_log=None):
        super().__init__(policy, env, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, train_freq=train_freq, gamma=gamma, tau=tau, tensorboard_log=tensorboard_log)
        self.num_heads = num_heads
        self.replay_buffer = BootstrappedReplayBuffer(buffer_size, env.observation_space, env.action_space, self.device, num_heads=num_heads)
        self.target_policy = BootstrappedDQNQNetwork(env.observation_space, env.action_space, policy.net_arch, policy.features_extractor, policy.features_dim, num_heads=num_heads)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        self.update(batch)

    def update(self, batch):
        obs, actions, rewards, next_obs, dones, head_masks = batch["observations"], batch["actions"], batch["rewards"], batch["next_observations"], batch["dones"], batch["head_masks"]
        loss = 0
        for head_idx in range(self.num_heads):
            mask = head_masks[:, head_idx]
            if mask.sum() == 0:
                continue
            q_values = self.policy.forward(obs, head_idx).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            next_q_values = self.target_policy.forward(next_obs, head_idx).max(dim=-1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss += (F.mse_loss(q_values[mask], target_q_values[mask]))
        loss /= self.num_heads
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        polyak_update(self.policy.parameters(), self.target_policy.parameters(), self.tau)
        
        # Log to TensorBoard
        if self.logger is not None:
            self.logger.record("train/loss", loss.item())
    
    def collect_rollouts(self, env, n_steps):
        obs = env.reset()
        for _ in range(n_steps):
            action = self.policy.select_action(obs, exploration=True)
            next_obs, reward, done, _ = env.step(action)
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs if not done else env.reset()

    def learn(self, total_timesteps):
        timesteps = 0
        while timesteps < total_timesteps:
            self.collect_rollouts(self.env, self.train_freq)
            self.train()
            timesteps += self.train_freq


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

    net_arch = [64, 64]
    obs_space = get_obs_shape(env.observation_space)
    policy = BootstrappedDQNQNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=net_arch,
        features_extractor=MlpExtractor(obs_space[0], net_arch,activation_fn=nn.ReLU,device=device),
        features_dim=net_arch[-1],  # Last layer's output dimension
        num_heads=10
    )
    model = BootstrappedDQN(
        policy=policy,
        env=env,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
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
