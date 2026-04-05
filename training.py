import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork
from castle.algorithms import Notears


def train_hill_climbing(data: pd.DataFrame) -> DiscreteBayesianNetwork:
    """Learn a Bayesian network structure using Hill Climbing search (pgmpy).

    Args:
        data: Training dataset as a DataFrame.

    Returns:
        A fitted BayesianNetwork with the learned DAG structure.
    """
    estimator = HillClimbSearch(data)
    dag = estimator.estimate(scoring_method="bic-g")
    model = DiscreteBayesianNetwork(dag.edges())
    model.fit(data)
    return model


def train_notears(data: pd.DataFrame):
    """Learn a DAG structure using NOTEARS (gcastle).

    Args:
        data: Training dataset as a DataFrame.

    Returns:
        A fitted Notears object. The learned adjacency matrix is available
        via ``model.causal_matrix``.
    """
    model = Notears()
    model.learn(data.to_numpy())
    return model
