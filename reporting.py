import matplotlib.pyplot as plt
import networkx as nx
import textwrap

# performance metrics for which the value is better when its lower
LOWER_IS_BETTER = {"shd"}

METRIC_COLUMNS = {
    "n_learned",
    "shd",
    "shd_normalized",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "reversed",
}


def plot_dag(dag: nx.DiGraph, title: str = "DAG") -> None:
    """Draw a DAG as an interactive pyvis graph rendered inline in Jupyter."""
    if not isinstance(dag, nx.DiGraph):
        new_dag = nx.DiGraph()
        new_dag.add_edges_from(dag)
        dag = new_dag
    pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
    labels = {n: textwrap.fill(str(n), width=11) for n in dag.nodes()}

    fig, ax = plt.subplots(figsize=(16, 10))
    nx.draw_networkx_edges(
        dag,
        pos,
        ax=ax,
        arrowstyle="-|>",
        arrowsize=13,
        width=1.1,
        edge_color="#666666",
        connectionstyle="arc3,rad=0.06",
        min_source_margin=16,
        min_target_margin=24,
    )
    nx.draw_networkx_labels(
        dag,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="#10323a",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#bfe3ef",
            edgecolor="#2b6777",
            linewidth=1.3,
        ),
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def graph_metrics(true_edges: set, learned_edges: set) -> dict:
    """Compute directed-edge recovery metrics between two edge sets.

    Reversed edges (u->v learned when v->u is true) count as 1 in SHD
    and are not counted as TP for either precision or recall."""

    reversed_edges = {(v, u) for u, v in true_edges} & learned_edges

    tp = len(true_edges & learned_edges)
    fp = len(learned_edges - true_edges - reversed_edges)
    fn = len(true_edges - learned_edges - {(v, u) for u, v in reversed_edges})
    rev = len(reversed_edges)

    shd = fp + fn + rev
    shd_normalized = shd / len(true_edges) if len(true_edges) > 0 else 0.0
    precision = tp / (tp + fp + rev) if (tp + fp + rev) > 0 else 0.0
    recall = tp / (tp + fn + rev) if (tp + fn + rev) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    # no true-negative notion without the full node-pair space, so accuracy is
    # the share of correctly recovered edges among all edges either true or learned
    accuracy = tp / (tp + fp + fn + rev) if (tp + fp + fn + rev) > 0 else 0.0

    return {
        "shd": shd,
        "shd_normalized": shd_normalized,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reversed": rev,
    }


def plot_hyperparam_sensitivity(results_df, metric="shd"):
    param_cols = [c for c in results_df.columns if c not in METRIC_COLUMNS]

    fig, axes = plt.subplots(1, len(param_cols), figsize=(5 * len(param_cols), 4))
    if len(param_cols) == 1:
        axes = [axes]

    for ax, param in zip(axes, param_cols):
        mean_per_value = results_df.groupby(param)[metric].mean()
        ax.plot(mean_per_value.index, mean_per_value.values, marker="o")
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {param}")

    fig.tight_layout()
    plt.show()


def print_summary(results_df, n=5, metric: str = "shd"):
    """Print the top-n combinations by the target metric and state the recommendation."""

    param_names = [c for c in results_df.columns if c not in METRIC_COLUMNS]
    # param_names = list(param_grid.keys())
    ascending = metric in LOWER_IS_BETTER
    top = results_df.sort_values(metric, ascending=ascending).head(n)
    cols = param_names + ["f1", "precision", "recall", "shd", "n_learned"]
    print("=" * 65)
    print(f"Top {n} combinations by {metric}")
    print("=" * 65)
    print(top[cols].round(3).to_string(index=False))
    best = top.iloc[0]
    params_str = ", ".join(f"{p}={best[p]}" for p in param_names)
    print(f"\nRecommended: {params_str}")
    print(
        f"  F1={best['f1']:.3f}  Precision={best['precision']:.3f}"
        f"  Recall={best['recall']:.3f}  SHD={best['shd']:.0f}"
    )
