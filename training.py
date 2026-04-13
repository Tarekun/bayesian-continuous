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
    """Learn a DAG structure using Hill Climbing with BIC-Gaussian scoring (pgmpy)."""

    def _learn(data):
        estimator = HillClimbSearch(data)
        dag = estimator.estimate(scoring_method="bic-g")
        return set(dag.edges())

    edges, elapsed = _timed(_learn, data)
    return TrainingResult(edges=edges, elapsed_s=elapsed)


def train_fges(data: pd.DataFrame, **kwargs) -> TrainingResult:
    """Learn a DAG structure using Greedy Equivalence Search / FGES (gcastle).

    Uses gcastle's GES implementation (Chickering, 2002; Ramsey et al., 2017).
    GES searches over CPDAGs, so some edge orientations may be undetermined by
    the data; the returned edges reflect the orientations present in the output.

    Keyword arguments are forwarded to ``castle.algorithms.GES``
    (e.g. ``criterion``, ``method``).
    For continuous data only ``criterion='bic'`` is supported; within BIC,
    ``method`` selects the scoring variant (``'scatter'`` or ``'r2'``).
    """

    def _learn(data):
        model = GES(**kwargs)
        model.learn(data.to_numpy())
        return _adj_to_edges(model.causal_matrix, list(data.columns))

    edges, elapsed = _timed(_learn, data)
    return TrainingResult(edges=edges, elapsed_s=elapsed)


def train_pc(
    data: pd.DataFrame,
    ci_test: str = "pearsonr",
    significance_level: float = 0.05,
    max_cond_vars: int | None = None,
) -> TrainingResult:
    """Learn a DAG structure using the PC algorithm (pgmpy).

    Args:
        data:               Training dataset.
        ci_test:            Conditional independence test to use
                            (e.g. ``"pearsonr"``, ``"chi_square"``, ``"g_sq"``).
        significance_level: Alpha threshold for the CI tests (default 0.05).
        max_cond_vars:      Maximum size of the conditioning set. ``None``
                            imposes no limit (full PC skeleton phase).

    Returns only the directed edges from the learned PDAG; undirected edges
    (where orientation is not identifiable from the data) are excluded.
    """

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


def train_notears(data: pd.DataFrame, **kwargs) -> TrainingResult:
    """Learn a DAG structure using NOTEARS (gcastle).

    Keyword arguments are forwarded to ``castle.algorithms.Notears``
    (e.g. ``lambda1``, ``w_threshold``, ``max_iter``, ``h_tol``, ``rho_max``).
    """

    def _learn(data):
        model = Notears(**kwargs)
        model.learn(data.to_numpy())
        return _adj_to_edges(model.causal_matrix, list(data.columns))

    edges, elapsed = _timed(_learn, data)
    return TrainingResult(edges=edges, elapsed_s=elapsed)


def learn_dag(
    data: pd.DataFrame,
    algorithm: Literal["hill_climbing", "pc", "fges", "notears"],
    **kwargs,
) -> TrainingResult:
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
