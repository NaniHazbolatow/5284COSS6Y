import numpy as np
from numba import njit

from CAStockModel.model.utils.constants import (BUY_PHASE_THD, SELL_PHASE_THD, CLUSTER_QUARTILE_FILTER_THD,
                                                CLUSTER_TOP_FILTER_THD, CLUSTER_THRESHOLD_FILTER_THD)
from CAStockModel.model.utils.utility_elements import order_clusters


def calc_singal_value(grid: np.ndarray, clusters: np.ndarray, cluster_filter_type: str) -> [float, int]:
    # Get the biggest (and therefore most relevant) clusters
    ordered_clusters = order_clusters(clusters)
    match cluster_filter_type:
        case "quantile":
            filtered_clusters = _filter_cluster_by_quartile(ordered_clusters)
        case "top":
            filtered_clusters = _filter_clusters_by_top_x(ordered_clusters)
        case "threshold":
            filtered_clusters = _filter_clusters_by_threshold(ordered_clusters)
        case _:
            raise ValueError(f"Filter type is not correct: {cluster_filter_type}")

    # Get traders which are in the biggest clusters
    traders = grid[np.any(filtered_clusters, axis=0)]

    # Calculate the signal value
    signal_value = _calc_signal_function_value(traders)

    return signal_value


@njit
def _filter_cluster_by_quartile(ordered_clusters: np.ndarray) -> np.ndarray:
    """Gets clusters in the top X quantile"""
    cutoff_idx = int(len(ordered_clusters) * CLUSTER_QUARTILE_FILTER_THD)
    return ordered_clusters[cutoff_idx:]


@njit
def _filter_clusters_by_top_x(ordered_clusters: np.ndarray) -> np.ndarray:
    """Gets the top X clusters"""
    if len(ordered_clusters) <= CLUSTER_TOP_FILTER_THD:
        return ordered_clusters[-CLUSTER_TOP_FILTER_THD:]
    return ordered_clusters


@njit
def _filter_clusters_by_threshold(ordered_clusters: np.ndarray) -> np.ndarray:
    """Gets every clusters above or equal to threshold X"""
    return ordered_clusters[ordered_clusters >= CLUSTER_THRESHOLD_FILTER_THD]


@njit
def _calc_signal_function_value(traders: np.ndarray) -> float:
    """Calculates signal value"""
    n_seller = np.sum(traders == -1)
    n_buyer = np.sum(traders == 1)
    return (n_buyer - n_seller) / (n_buyer + n_seller)


@njit
def det_phase(signal_value: float) -> int:
    """Determines the phase based on the signal value"""
    if signal_value >= BUY_PHASE_THD:
        return 1
    elif signal_value <= SELL_PHASE_THD:
        return -1
    else:
        return 0
