"""
Novelty Sampler
"""
from typing import Iterable, Literal, Optional

import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler

from autora.utils.deprecation import deprecated_alias

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
    reference_conditions: np.ndarray,
    num_samples: Optional[int] = None,
    metric: AllowedMetrics = "euclidean",
    integration: str = "min",
) -> np.ndarray:
    """
    This novelty sampler re-arranges the pool of experimental conditions according to their
    dissimilarity with respect to a reference pool. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in the pool and the reference conditions.
    If no number of samples are specified, all samples will be ordered and returned from the pool.

    Args:
        condition_pool: pool of experimental conditions to evaluate dissimilarity
        reference_conditions: reference pool of experimental conditions
        num_samples: number of samples to select from the pool of experimental conditions (the default is to select all)
        metric (str): dissimilarity measure. Options: 'euclidean', 'manhattan', 'chebyshev',
            'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
            'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice',
            'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
            'sokalsneath', 'yule'. See [sklearn.metrics.DistanceMetric][] for more details.

    Returns:
        Sampled pool of conditions
    """

    new_conditions, distance_scores = novelty_score_sampler(condition_pool, reference_conditions, num_samples, metric, integration)

    return new_conditions


def novelty_score_sample(
    condition_pool: np.ndarray,
    reference_conditions: np.ndarray,
    num_samples: Optional[int] = None,
    metric: AllowedMetrics = "euclidean",
    integration: str = "sum",
) -> np.ndarray:
    """
    This dissimilarity samples re-arranges the pool of experimental conditions according to their
    dissimilarity with respect to a reference pool. The default dissimilarity is calculated
    as the average of the pairwise distances between the conditions in the pool and the reference conditions.
    If no number of samples are specified, all samples will be ordered and returned from the pool.

    Args:
        condition_pool: pool of experimental conditions to evaluate dissimilarity
        reference_conditions: reference pool of experimental conditions
        num_samples: number of samples to select from the pool of experimental conditions (the default is to select all)
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

    if isinstance(condition_pool, Iterable):
        condition_pool = np.array(list(condition_pool))

    if isinstance(reference_conditions, Iterable):
        reference_conditions = np.array(list(reference_conditions))

    if condition_pool.ndim == 1:
        condition_pool = condition_pool.reshape(-1, 1)

    if reference_conditions.ndim == 1:
        reference_conditions = reference_conditions.reshape(-1, 1)

    if condition_pool.shape[1] != reference_conditions.shape[1]:
        raise ValueError(
            f"condition_pool and reference_conditions must have the same number of columns.\n"
            f"condition_pool has {condition_pool.shape[1]} columns, while reference_conditions has {reference_conditions.shape[1]} columns."
        )

    if num_samples is None:
        num_samples = condition_pool.shape[0]

    if condition_pool.shape[0] < num_samples:
        raise ValueError(
            f"condition_pool must have at least {num_samples} rows matching the number of requested samples."
        )

    dist = DistanceMetric.get_metric(metric)

    distances = dist.pairwise(reference_conditions, condition_pool)

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
    sorted_condition_pool = condition_pool[np.argsort(integrated_distance)[::-1]]
    sorted_score = score[np.argsort(score)[::-1]]

    return sorted_condition_pool[:num_samples], sorted_score[:num_samples]

novelty_score_sampler = deprecated_alias(novelty_score_sample, "novelty_score_sampler")
