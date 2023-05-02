"""
Novelty Sampler
"""
from typing import Iterable, Literal

import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler

AllowedMetrics = Literal[
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "wminkowski",
    "seuclidean",
    "mahalanobis",
    "haversine",
    "hamming",
    "canberra",
    "braycurtis",
    "matching",
    "jaccard",
    "dice",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
]


def novelty_sampler(
    condition_pool: np.ndarray,
    num_samples: int,
    reference_conditions: np.ndarray,
    metric: AllowedMetrics = "euclidean",
    integration: str = "min",
) -> np.ndarray:
    """
    This dissimilarity samples re-arranges the pool of experimental conditions according to their
    dissimilarity with respect to a reference pool. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in the pool and the reference conditions.

    Args:
        condition_pool: pool of IV conditions to evaluate dissimilarity
        num_samples: number of samples to select
        reference_conditions: reference pool of IV conditions
        metric (str): dissimilarity measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See [sklearn.metrics.DistanceMetric][] for more details.

    Returns:
        Sampled pool of conditions
    """

    new_conditions, distance_scores = compute_dissimilarity(condition_pool, num_samples, reference_conditions, metric, integration)

    return new_conditions


def compute_dissimilarity(
    condition_pool: np.ndarray,
    num_samples: int,
    reference_conditions: np.ndarray,
    metric: AllowedMetrics = "euclidean",
    integration: str = "sum",
) -> np.ndarray:
    """
    This dissimilarity samples re-arranges the pool of experimental conditions according to their
    dissimilarity with respect to a reference pool. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in the pool and the reference conditions.

    Args:
        condition_pool: pool of IV conditions to evaluate dissimilarity
        num_samples: number of samples to select
        reference_conditions: reference pool of IV conditions
        metric (str): dissimilarity measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See [sklearn.metrics.DistanceMetric][] for more details.
        integration: Distance integration method used to compute the overall dissimilarity score
        for a given data point. Options: 'sum', 'prod', 'mean', 'min', 'max'.

    Returns:
        Sampled pool of conditions and dissimilarity scores
    """

    X = condition_pool
    X_ref = reference_conditions

    if isinstance(X, Iterable):
        X = np.array(list(X))

    if isinstance(X_ref, Iterable):
        X_ref = np.array(list(X_ref))

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)

    if X.shape[1] != X_ref.shape[1]:
        raise ValueError(
            f"X and X_ref must have the same number of columns.\n"
            f"X has {X.shape[1]} columns, while X_ref has {X_ref.shape[1]} columns."
        )

    if X.shape[0] < num_samples:
        raise ValueError(
            f"X must have at least {num_samples} rows matching the number of requested samples."
        )

    dist = DistanceMetric.get_metric(metric)

    distances = dist.pairwise(X_ref, X)

    if integration == "sum":
        integrated_distance = np.sum(distances, axis=0)
    elif integration == "mean":
        integrated_distance = np.mean(distances, axis=0)
    elif integration == "max":
        integrated_distance = np.max(distances, axis=0)
    elif integration == "min":
        integrated_distance = np.min(distances, axis=0)
    elif integration == "prod":
        integrated_distance = np.prod(distances, axis=0)
    else:
        raise ValueError(f"Integration method {integration} not supported.")

    # normalize the distances
    scaler = StandardScaler()
    score = scaler.fit_transform(integrated_distance.reshape(-1, 1)).flatten()

    # order rows in Y from highest to lowest
    sorted_X = X[np.argsort(integrated_distance)[::-1]]
    sorted_score = score[np.argsort(score)[::-1]]

    return sorted_X[:num_samples], sorted_score[:num_samples]
