import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.causal_discovery import PC
from castle.algorithms import Notears, GES


@dataclass
class TrainingResult:
    edges: set[tuple[str, str]]
    elapsed_s: float


def _timed(fn, *args, **kwargs) -> tuple:
    """Runs `fn` on `*args,**kwargs` timing how long it takes.
    Returns a tuple containing (result, time)"""

    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _adj_to_edges(causal_matrix, columns: list[str]) -> set[tuple[str, str]]:
    n = len(columns)
    return {
        (columns[i], columns[j])
        for i in range(n)
        for j in range(n)
        if causal_matrix[i, j] != 0
    }


def train_hill_climbing(data: pd.DataFrame) -> TrainingResult:
    def _learn(data):
        estimator = HillClimbSearch(data)
        dag = estimator.estimate(scoring_method="bic-g")
        return set(dag.edges())

    edges, elapsed = _timed(_learn, data)
    return TrainingResult(edges=edges, elapsed_s=elapsed)


def train_fges(
    data: pd.DataFrame,
    # GES parameters
    criterion: str = "bic",
    method: str = "scatter",
    k: float = 0.001,
    N: int = 10,
) -> TrainingResult:
    def _learn(data):
        model = GES(
            criterion=criterion,
            method=method,
            k=k,
            N=N,
        )
        model.learn(data.to_numpy())
        return _adj_to_edges(model.causal_matrix, list(data.columns))

    edges, elapsed = _timed(_learn, data)
    return TrainingResult(edges=edges, elapsed_s=elapsed)


def train_pc(
    data: pd.DataFrame,
    # PC parameters
    ci_test: str = "pearsonr",
    significance_level: float = 0.01,
    max_cond_vars: int | None = None,
) -> TrainingResult:
    def _learn(data):
        kwargs = dict(
            ci_test=ci_test,
            significance_level=significance_level,
            return_type="pdag",
            show_progress=False,
        )
        if max_cond_vars is not None:
            kwargs["max_cond_vars"] = max_cond_vars
        model = PC(**kwargs)
        model.fit(data)
        return set(model.causal_graph_.edges())

    edges, elapsed = _timed(_learn, data)
    return TrainingResult(edges=edges, elapsed_s=elapsed)


def train_notears(
    data: pd.DataFrame,
    # NOTEARS parameters
    lambda1: float = 0.1,
    loss_type: str = "l2",
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 10000000000000000,
    w_threshold: float = 0.3,
) -> TrainingResult:
    def _learn(data):
        model = Notears(
            lambda1=lambda1,
            loss_type=loss_type,
            max_iter=max_iter,
            h_tol=h_tol,
            rho_max=rho_max,
            w_threshold=w_threshold,
        )
        model.learn(data.to_numpy())
        return _adj_to_edges(model.causal_matrix, list(data.columns))

    edges, elapsed = _timed(_learn, data)
    return TrainingResult(edges=edges, elapsed_s=elapsed)


def learn_dag(
    data: pd.DataFrame,
    algorithm: Literal["hill_climbing", "pc", "fges", "notears"],
    **kwargs,
) -> TrainingResult:
    """Generic entrypoint to learn a Bayesian network using one of the supported algorithms"""

    dispatch: dict[str, callable] = {
        "hill_climbing": train_hill_climbing,
        "pc": train_pc,
        "fges": train_fges,
        "notears": train_notears,
    }
    if algorithm not in dispatch:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Choose from {list(dispatch)}."
        )
    return dispatch[algorithm](data, **kwargs)
