import numpy as np
from scipy.ndimage import label


def initialise_market_grid(shape: tuple, actice_ratio: float) -> np.ndarray:
    active_indices = np.random.choice(shape[0] * shape[1], replace=False, size=int(actice_ratio * shape[0] * shape[1]))

    one_d_grid = np.zeros(shape=shape[0] * shape[1], dtype=np.int8)
    one_d_grid[active_indices] = np.random.choice([1, -1], replace=True, size=len(active_indices))

    two_d_grid = one_d_grid.reshape(shape)

    return two_d_grid


def find_clusters(grid: np.ndarray) -> np.ndarray:
    active_cells = (grid == 1) | (grid == -1)

    von_neumann_structure = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]])

    clusters, num_clusters = label(active_cells, structure=von_neumann_structure)

    cluster_filters = np.array([(clusters == cluster_id) for cluster_id in range(1, num_clusters + 1)])

    return cluster_filters


def order_clusters(clusters: np.ndarray) -> np.ndarray:
    order = np.argsort(np.sum(clusters, axis=(1, 2)))
    return clusters[order]



def calculate_log_return(grid: np.ndarray, clusters: np.ndarray, beta: float) -> float:

    trader_weighted_sum = np.sum([np.sum(cluster) * np.sum(grid[cluster]) for cluster in clusters])
    log_return = trader_weighted_sum / beta

    return log_return


def convert_time(elapsed_time: int) -> str:
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours} h {minutes} m {seconds} s"
