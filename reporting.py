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
    precision = tp / (tp + fp + rev) if (tp + fp + rev) > 0 else 0.0
    recall = tp / (tp + fn + rev) if (tp + fn + rev) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "shd": shd,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reversed": rev,
    }
