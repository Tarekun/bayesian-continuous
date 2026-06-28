from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from typing import Literal

SupportedDistribution = Literal["gaussian"]


def build_random_dag(
    n_nodes: int, edge_prob: float = 0.3, seed: int | None = None
) -> nx.DiGraph:
    """Build a random DAG over ``n_nodes`` ordered nodes.

    Args:
        n_nodes: Number of nodes. Nodes are named ``X0, X1, ..., X{n-1}``.
        edge_prob: Probability of including each candidate edge (i -> j) for i < j.
        seed: Optional random seed for reproducibility.

    Returns:
        A ``networkx.DiGraph`` whose nodes are ``["X0", "X1", ..., "X{n-1}"]``
        and whose edges form a DAG.
    """
    rng = np.random.default_rng(seed)
    nodes = [f"X{i}" for i in range(n_nodes)]
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                dag.add_edge(nodes[i], nodes[j])
    return dag


def generate_random_dataset(
    n_features: int,
    n_samples: int,
    distribution: SupportedDistribution = "gaussian",
    ground_truth_dag: nx.DiGraph | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a random dataset with the given number of features and samples.

    When ``ground_truth_dag`` is provided the data is sampled from a **linear
    Gaussian SEM** (Structural Equation Model) whose graph structure matches
    the DAG.
    When no DAG is given, all features are sampled i.i.d. from the chosen
    marginal distribution (no causal structure imposed).
    """

    if distribution != "gaussian":
        raise ValueError(f"Unsupported distribution: {distribution!r}")

    rng = np.random.default_rng(seed)

    if ground_truth_dag is not None:
        return _sample_from_dag(ground_truth_dag, n_features, n_samples, rng)
    else:
        data = rng.standard_normal((n_samples, n_features))
        columns = [f"X{i}" for i in range(n_features)]
        return pd.DataFrame(data, columns=columns)


def _sample_from_dag(
    dag: nx.DiGraph,
    n_features: int,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample ``n_samples`` rows from a linear Gaussian SEM defined by ``dag``.  Each variable ``Xv`` is generated as:

        Xv = sum_{p in parents(Xv)} w_{p,v} * Xp  +  eps_v,   eps_v ~ N(0, 1)

    where the edge weights ``w_{p,v}`` are drawn uniformly from
    ``[-1, -0.5] ∪ [0.5, 1]`` to avoid near-zero effects.  Variables are
    generated in topological order, so every parent is available before its
    children."""

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("ground_truth_dag must be a directed acyclic graph (DAG).")

    nodes = list(dag.nodes)
    if len(nodes) != n_features:
        raise ValueError(
            f"ground_truth_dag has {len(nodes)} nodes but n_features={n_features}."
        )

    # assign edge weights drawn from [-1, -0.5] ∪ [0.5, 1]
    weights: dict[tuple[str, str], float] = {}
    for u, v in dag.edges():
        w = rng.uniform(0.5, 1.0)
        if rng.random() < 0.5:
            w = -w
        weights[(u, v)] = w

    # generate data column-by-column in topological order
    values: dict[str, np.ndarray] = {}
    for node in nx.topological_sort(dag):
        noise = rng.standard_normal(n_samples)
        parents = list(dag.predecessors(node))
        if parents:
            parent_contribution = sum(weights[(p, node)] * values[p] for p in parents)
            values[node] = parent_contribution + noise
        else:
            values[node] = noise

    return pd.DataFrame(values, columns=nodes)


_SACHS_DIR = Path(__file__).parent / "datasets" / "sachs"


def load_sachs_data() -> pd.DataFrame:
    """Load the Sachs et al. (2005) protein signaling dataset."""

    data_dir = _SACHS_DIR / "Data Files"
    frames = [pd.read_csv(f) for f in sorted(data_dir.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


def load_sachs_ground_truth() -> nx.DiGraph:
    """Load the Sachs et al. (2005) ground-truth causal network."""

    gt_df = pd.read_csv(_SACHS_DIR / "GroundTruth.csv")
    dag = nx.DiGraph()
    for _, row in gt_df.iterrows():
        dag.add_edge(row["from"], row["to"])
    return dag
