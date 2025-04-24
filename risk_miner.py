import json
import os
import glob
from typing import Optional, Tuple, List, Any
from datetime import datetime
from pathlib import Path
from openai import OpenAI

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import utils
from torch.utils.tensorboard import SummaryWriter

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

def load_linear_alpha_pool_from_json(json_path: str,
                                     capacity: int,
                                     calculator: QLibStockDataCalculator,
                                     single_alpha: bool = False) -> LinearAlphaPool | list[LinearAlphaPool]:
    # Load the JSON file
    parser = ExpressionParser(Operators)
    with open(json_path, 'r') as f:
        pool_data = json.load(f)

    # Extract expressions and weights from the loaded data
    expressions = pool_data['exprs']
    weights = pool_data['weights']

    # Create an instance of LinearAlphaPool
    alpha_pool = MseAlphaPool(
        capacity=capacity,  # Set the capacity based on the number of expressions
        calculator=calculator,
        ic_lower_bound=None,
        l1_alpha=5e-3,
    )

    # Load the expressions into the pool
    expres = []
    if single_alpha:
        alpha_pools = []

        for expression,weight in zip(expressions,weights):
            alpha_pool = MseAlphaPool(
                capacity=1,
                calculator=calculator
                )
            expre = parser.parse(expression)
            alpha_pool.force_load_exprs([expre], [weight])
            alpha_pools.append(alpha_pool)

        return  alpha_pools
    else:
        for expression in expressions:
            expre = parser.parse(expression)
            expres.append(expre)
        
        
        alpha_pool.force_load_exprs(expres, weights)

        return alpha_pool


class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        tb_log_name: str = "riskminer_tensorboard",
        drop_rl_n: int = 5
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        self.tb_log_name = tb_log_name
        os.makedirs(self.save_path, exist_ok=True)

        # Create a SummaryWriter that does NOT purge the log (resume mode)
        log_dir = f"./out/riskminer_tensorboard/{tb_log_name}"
        self.writer = SummaryWriter(log_dir=log_dir)
        
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

        # Instead of self.logger.record, we use self.writer.add_scalar
        self.writer.add_scalar('pool/size', self.pool.size, self.num_timesteps)
        self.writer.add_scalar('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum(), self.num_timesteps)
        self.writer.add_scalar('pool/best_ic_ret', self.pool.best_ic_ret, self.num_timesteps)
        self.writer.add_scalar('pool/eval_cnt', self.pool.eval_cnt, self.num_timesteps)
        self.writer.add_scalar('pool/expr_len', len(self.env_core._tokens), self.num_timesteps)

        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.writer.add_scalar(f'test/ic_{i}', ic_test, self.num_timesteps)
            self.writer.add_scalar(f'test/rank_ic_{i}', rank_ic_test, self.num_timesteps)
        self.writer.add_scalar('test/ic_mean', ic_test_mean, self.num_timesteps)
        self.writer.add_scalar('test/rank_ic_mean', rank_ic_test_mean, self.num_timesteps)

        self.save_checkpoint()

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        self.locals = locals_
        self.globals = globals_
        # Resume num_timesteps if available
        algo = locals_.get("self", None)
        if algo is not None and hasattr(algo, "num_timesteps"):
            self.num_timesteps = algo.num_timesteps
        else:
            self.num_timesteps = 0
        if self.verbose:
            print(f"Resuming training from timestep: {self.num_timesteps}")
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
    # os.makedirs(save_path, exist_ok=True)

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
    
    policy_net_kwargs = {
        "input_dim": env.observation_space.shape[0],  # dimension of observation
        "hidden_dim": 64,                              # GRU hidden dimension
        "num_layers": 4,                               # number of GRU layers
        "action_dim": env.action_space.n,              # number of actions
        "mlp_hidden_sizes": [32, 32],                  # MLP hidden layers
        "lr": 0.001                                    # learning rate
    }
    

    """
    Resume training or new.
    """

    # resume = False
    resume = True

    if resume:

        all_subdirs = os.listdir('./out/risk_miner')
        latest_subdir = all_subdirs[-1]
        list_of_files = glob.glob(f'./out/risk_miner/{latest_subdir}/*.pt')

        latest_file = max(list_of_files, key=os.path.getctime)
        latest_step = int(latest_file.split("\\")[-1].split("_")[0])

        list_of_pools = glob.glob(f'./out/risk_miner/{latest_subdir}/*.json')
        latest_pool = max(list_of_pools, key=os.path.getctime)

        latest_pool = load_linear_alpha_pool_from_json(latest_pool,capacity=pool_capacity, calculator=calculators[0])

        print(f"Resuming training from step {latest_step}...")
        checkpoint_path = latest_file.replace("\\", "/")
        name_prefix = os.listdir('out/riskminer_tensorboard')[-1]
        save_path = f'./out/risk_miner/{latest_subdir}'

        checkpoint_callback = CustomCallback(
            save_path=save_path,
            test_calculators=calculators[1:],
            verbose=1,
            chat_session=inter,
            llm_every_n_steps=llm_every_n_steps,
            drop_rl_n=drop_rl_n,
            tb_log_name=name_prefix
        )

        # checkpoint_callback.pool = latest_pool
        env = AlphaEnvDense(
            pool=latest_pool,
            device=device,
            print_expr=True,
            # penalty = True,
            # constrain = True,
            # her = True
        )

        model = RiskMCTSAlgorithm.load(checkpoint_path, env, policy_net_kwargs)
        model.learn(
            total_timesteps= steps - model.num_timesteps,
            callback=checkpoint_callback,         # You can add a callback if desired.
            log_interval=10,
            progress_bar=True
        )
    else:
        print("Starting new training...")
        os.makedirs(save_path, exist_ok=True)

        ## Vanila MCTS

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

        checkpoint_callback = CustomCallback(
            save_path=save_path,
            test_calculators=calculators[1:],
            verbose=1,
            chat_session=inter,
            llm_every_n_steps=llm_every_n_steps,
            drop_rl_n=drop_rl_n,
            tb_log_name=name_prefix
        )

        model = RiskMCTSAlgorithm(
            env=env,
            policy_net_kwargs=policy_net_kwargs,
            n_simulations=20,   # number of MCTS simulations per planning step
            c_param=1.41,
            alpha=0.7,          # risk-seeking quantile level
            replay_size=10000,   # smaller replay size for demo purposes
            batch_size=128,
            tensorboard_log="./out/riskminer_tensorboard",
            gamma=1.0
        )

        model.learn(
            total_timesteps=steps,
            callback=checkpoint_callback,
            log_interval=10,
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