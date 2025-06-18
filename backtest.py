from typing import Optional, TypeVar, Callable, Optional, Tuple
import os
import pickle
import warnings
import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.strategy import TopkDropoutStrategy
# from qlib.data import D

from alphagen.data.expression import *
from alphagen.data.parser import parse_expression
from alphagen_generic.features import *
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.utils import load_alpha_pool_by_path


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj
def normalize_series(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()

def rank_series_per_date(series: pd.Series) -> pd.Series:
    """
    Rank the series for each date (assumed to be the first level of the MultiIndex).
    The highest value is assigned rank 1.
    """
    return series.groupby(level=0).rank(ascending=False, method='min')
def compute_rmse_per_date(model_scores: pd.Series, oracle_scores: pd.Series) -> pd.DataFrame:
    """
    Compute the RMSE across stocks for each date.
    
    Parameters:
      model_scores: pd.Series with MultiIndex (date, instrument) containing your model's prediction scores.
      oracle_scores: pd.Series with MultiIndex (date, instrument) containing the oracle's prediction scores.
      
    Returns:
      A DataFrame with the date as the index and a column 'rmse' containing the RMSE for that date.
    """
    # normalize
    # model_scores = normalize_series(model_scores)
    # oracle_scores = normalize_series(oracle_scores)

    # rank the scores
    model_scores = rank_series_per_date(model_scores)
    oracle_scores = rank_series_per_date(oracle_scores)

    # Combine both series into one DataFrame
    df = pd.DataFrame({
        "model": model_scores,
        "oracle": oracle_scores
    })
    # Group by the date level. If your MultiIndex doesn't have names,
    # you can group by level=0 (assuming the first level is the date).
    rmse_series = df.groupby(level=0).apply(
        lambda group: np.sqrt(((group["oracle"] - group["model"]) ** 2).mean())
    )
    rmse_df = rmse_series.to_frame(name="rmse")
    # Ensure the index is named "date" (or adjust as needed)
    rmse_df.index.name = "date"
    return rmse_df


@dataclass
class BacktestResult(DataClassJsonMixin):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 50,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: Union[pd.Series, pd.DataFrame],
        oracle_signal:Optional[pd.DataFrame] = None,
        output_prefix: Optional[str] = None
    ) -> Tuple[pd.DataFrame, BacktestResult]:
        
        # calculate RMSE
        if oracle_signal is not None:
            rmse_df = compute_rmse_per_date(prediction.iloc[:,0], oracle_signal.iloc[:,0])

            dump_pickle(output_prefix + "-rmse.pkl", lambda: rmse_df, True)

        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy = TopkDropoutStrategy(
                    signal=prediction,
                    topk=self._top_k,
                    n_drop=self._n_drop,
                    only_tradable=True,
                    forbid_all_trade_at_limit=True
                )
                executor = exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", result.to_json())
        return report, result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )


if __name__ == "__main__":
    initialize_qlib("~/.qlib/qlib_data/cn_data")
    qlib_backtest = QlibBacktest(top_k=50, n_drop=5)
    data = StockData(
        instrument="csi300",
        start_time="2022-07-01",
        end_time="2024-01-01"
    )
    calc = QLibStockDataCalculator(data, None)

    def run_backtest(prefix: str, seed: int, exprs: List[Expression], weights: List[float]):
        df = data.make_dataframe(calc.make_ensemble_alpha(exprs, weights))
        oracle_signal = data.compute_oracle_scores()
        qlib_backtest.run(df,oracle_signal=oracle_signal, output_prefix=f"out/backtests/51-5/{prefix}/{seed}")

    for p in Path("out/gp").iterdir():
        seed = int(p.name)
        with open(p / "40.json") as f:
            report = json.load(f)
        state = report["res"]["res"]["pool_state"]
        run_backtest("gp", seed, [parse_expression(e) for e in state["exprs"]], state["weights"])
    # exit(0)
    for p in Path("out/results").iterdir():
        inst, size, seed, time, ver = p.name.split('_', 4)
        size, seed = int(size), int(seed)
        if inst != "csi300" or size != 20 or time < "20240923" or ver == "llm_d5":
            continue
        exprs, weights = load_alpha_pool_by_path(str(p / "251904_steps_pool.json"))
        run_backtest(ver, seed, exprs, weights)

    for p in Path("out/boot_dqn").iterdir():
        inst, size, seed, time, ver = p.name.split('_', 4)
        size, seed = int(size), int(seed)
        if inst != "csi300" or size != 20 or time < "20240923" or ver == "llm_d5":
            continue
        try:
            exprs, weights = load_alpha_pool_by_path(str(p / "57700_steps_pool.json"))
        except:
            continue
        run_backtest(ver, seed, exprs, weights)

    for p in Path("out/risk_miner").iterdir():
        inst, size, seed, time, ver = p.name.split('_', 4)
        size, seed = int(size), int(seed)
        if inst != "csi300" or size != 20 or time < "20240923" or ver == "llm_d5":
            continue
        try:
            exprs, weights = load_alpha_pool_by_path(str(p / "12477_steps_pool.json"))
        except:
            continue
        run_backtest(ver, seed, exprs, weights)

    for p in Path("out/e_miner").iterdir():
        inst, size, seed, time, ver = p.name.split('_', 4)
        size, seed = int(size), int(seed)
        if inst != "csi300" or size != 20 or time < "20240923" or ver == "llm_d5":
            continue

        try:
            exprs, weights = load_alpha_pool_by_path(str(p / "496392_steps_pool.json"))
            print(f"Loaded {len(exprs)} expressions from {p.name}")
        except Exception as e:
            continue

        run_backtest(ver, seed, exprs, weights)

    # oracle back test

    oracle_signal = data.compute_oracle_scores()
    qlib_backtest.run(oracle_signal, output_prefix=f"out/backtests/51-5/oracle/0")
    exit(0)
    for p in Path("out/llm-tests/interaction").iterdir():
        if not p.name.startswith("v1"):
            continue
        run = int(p.name[3])
        with open(p / "report.json") as f:
            report = json.load(f)
        state = report[-1]["pool_state"]
        run_backtest("pure_llm", run, [parse_expression(t[0]) for t in state], [t[1] for t in state])
