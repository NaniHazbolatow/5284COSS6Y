import numpy as np
from numba import njit

from CAStockModel.model.utils.constants import (BUY_PHASE_THD, SELL_PHASE_THD, CLUSTER_QUARTILE_FILTER_THD,
                                                CLUSTER_TOP_FILTER_THD, CLUSTER_THRESHOLD_FILTER_THD)
from CAStockModel.model.utils.utility_elements import order_clusters


def calc_singal_value(grid: np.ndarray, clusters: np.ndarray, cluster_filter_type: str) -> [float, int]:
    """
    Calculates the signal value for a given grid. The value equals to the number of sellers subtracted from the number
    of buyers in the bigger clusters. Then this number is divided by the sum of these two numbers, normalizing it to
    the [-1, 1] range.
    With formula: (N_buy - N_sell) / (N_buy + N_sell)

    Args:
        grid: 2D numpy array, with values -1, 0, 1 representing the decisions of the traders
        clusters: 3D numpy array, in which every 2D array along the 0th axis is a filter for a cluster in the grid
        cluster_filter_type: The way how the top clusters should be chosen for signal value calculation

    Returns:
        Signal value
    """
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
    """
    Gets clusters in the top X quantile.

    Args:
        ordered_clusters: 3D numpy array, in which every 2D array along the 0th axis is a filter for a cluster in the
                            grid. The filters are in increasing order by the number of active traders (True values)

    Returns:
        Filters for the clusters in the top X quantile in a 3D array
    """
    cutoff_idx = int(len(ordered_clusters) * CLUSTER_QUARTILE_FILTER_THD)
    return ordered_clusters[cutoff_idx:]


@njit
def _filter_clusters_by_top_x(ordered_clusters: np.ndarray) -> np.ndarray:
    """
    Gets the top X clusters.

    Args:
        ordered_clusters: 3D numpy array, in which every 2D array along the 0th axis is a filter for a cluster in the
                           grid. The filters are in increasing order by the number of active traders (True values)

    Returns:
        Filters for the top X  clusters in a 3D array
    """
    if len(ordered_clusters) <= CLUSTER_TOP_FILTER_THD:
        return ordered_clusters[-CLUSTER_TOP_FILTER_THD:]
    return ordered_clusters


@njit
def _filter_clusters_by_threshold(ordered_clusters: np.ndarray) -> np.ndarray:
    """
    Gets every clusters above or equal to threshold X.

    Args:
        ordered_clusters: 3D numpy array, in which every 2D array along the 0th axis is a filter for a cluster in the
                           grid. The filters are in increasing order by the number of active traders. (True values)

    Returns:
        Filters for the clusters which has more traders than a threshold X in a 3D array.
    """
    return ordered_clusters[ordered_clusters >= CLUSTER_THRESHOLD_FILTER_THD]


@njit
def _calc_signal_function_value(traders: np.ndarray) -> float:
    """
    Calculates signal value.

    Args:
        traders: 1D numpy array consisting -1, 1 representing the traders and their decision.

    Returns:
        Signal value
    """
    n_seller = np.sum(traders == -1)
    n_buyer = np.sum(traders == 1)
    return (n_buyer - n_seller) / (n_buyer + n_seller)


@njit
def det_phase(signal_value: float) -> int:
    """
    Determines the phase based on the signal value.

    Args:
        signal_value: the result of the signal value calculation.

    Returns:
        Phase index
    """
    if signal_value >= BUY_PHASE_THD:
        return 1
    elif signal_value <= SELL_PHASE_THD:
        return -1
    else:
        return 0
