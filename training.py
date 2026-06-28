from castle.algorithms import Notears, GES
from dataclasses import dataclass
import itertools
import networkx as nx
import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.causal_discovery import PC
import time
from typing import Literal

from reporting import graph_metrics, LOWER_IS_BETTER


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


def grid_search(
    data,
    algorithm: Literal["hill_climbing", "pc", "fges", "notears"],
    param_grid: dict,
    gt_dag: nx.DiGraph,
    metric: str = "shd",
):
    """Grid search over all combinations in param_grid.
    For each combination the algorithm is run on `data` and the resulting
    edges are scored against `gt_dag`

    Args:
        data:       training dataset
        algorithm:  algorithm name forwarded to learn_dag
        param_grid: mapping of parameter name -> list of candidate values
        gt_dag:     ground-truth DAG
        metric:     metric used to select the best result ("shd", "f1",
                    "precision", or "recall"). defaults to "shd".

    Returns:
        best_result: TrainingResult for the combination with the best score on metric
        results_df:  DataFrame with one row per parameter combination,
                     columns = param names + [n_learned, shd, precision, recall, f1].
    """

    true_edges = set(gt_dag.edges())
    rows = []
    lower_is_better = metric in LOWER_IS_BETTER
    best_result = None
    best_val = float("inf") if lower_is_better else -1.0

    combos = 1
    for v in param_grid.values():
        combos *= len(v)
    print(
        f"Grid: {' × '.join(str(len(v)) for v in param_grid.values())} = {combos} combinations\n"
    )

    for param_values in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), param_values))
        result = learn_dag(data, algorithm, **params)
        m = graph_metrics(true_edges, result.edges)
        rows.append({**params, "n_learned": len(result.edges), **m})
        val = m[metric]
        if (val < best_val) if lower_is_better else (val > best_val):
            best_val, best_result = val, result

    return best_result, pd.DataFrame(rows)
