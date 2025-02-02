import numpy as np
from scipy.ndimage import label


def initialise_market_grid(shape: tuple, active_ratio: float) -> np.ndarray:
    """
    Initialises a grid (2D numpy array) representing the market. Every cell in this grid is going to take the number
    from the set {1, 0, -1}. The ratio of traders having 1 or -1 among all the traders is going to be equal to the
    predetermined value.

    Args:
        shape: Shape information of the market. (The multiplication of the two numbers results in the number of traders on the market.)
        active_ratio: Initial ratio of active traders (having either 1 or -1 value) among all the traders.

    Returns:
        Grid (2D numpy array) representing the market.
    """
    # Randomly choosing indices of for the active traders
    active_indices = np.random.choice(shape[0] * shape[1], replace=False, size=int(active_ratio * shape[0] * shape[1]))

    # Creating the flat version of the market and assigning the active traders on it with random decisions of buying or selling
    one_d_grid = np.zeros(shape=shape[0] * shape[1], dtype=np.int8)
    one_d_grid[active_indices] = np.random.choice([1, -1], replace=True, size=len(active_indices))

    # Reshaping the flat version of the market to get the regular 2D grid
    two_d_grid = one_d_grid.reshape(shape)

    return two_d_grid


def find_clusters(grid: np.ndarray) -> np.ndarray:
    """
    Creates a 3D numpy array, where every slice along the 0 axis is a 2D numpy array filled with boolean values
    filtering traders who are part of that clusters and those who are not. Therefore, every slice is going to represent
    a cluster. The order of the slices are random.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.

    Returns:
        3D numpy array consisting the filters representing clusters
    """
    # Create the filter for active traders
    active_cells = (grid == 1) | (grid == -1)

    # Define the neighbour structure that is used in cluster definition
    von_neumann_structure = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]])

    # Defining clusters
    clusters, num_clusters = label(active_cells, structure=von_neumann_structure)

    # Creating the 3D numpy array structure
    cluster_filters = np.array([(clusters == cluster_id) for cluster_id in range(1, num_clusters + 1)])

    return cluster_filters


def order_clusters(clusters: np.ndarray) -> np.ndarray:
    """
    Orders cluster input (3D numpy array) along the 0 axis based on the cluster sizes. The order is increasing.

    Args:
        clusters: 3D numpy array consisting the filters representing clusters along the 0 axis.

    Returns:
        3D numpy array consisting the filters representing clusters ordered by their sizes.
    """
    order = np.argsort(np.sum(clusters, axis=(1, 2)))
    return clusters[order]



def calculate_log_return(grid: np.ndarray, clusters: np.ndarray, beta: float) -> float:
    """
        Calculates log return based on the weighted market balance. Every active trader decision to sell or buy is going
        to be weighted by the size of the cluster they belong to. The weighted sum is then normalized to resulting in
        the final log return.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        clusters: 3D numpy array where every slice along the 0 axis consists of filter of traders belonging to the same clusters.
        beta: Normalising parameter.

    Returns:
        The calculated final log return.
    """
    # Calculating the weighted sum of the active traders' decisions on the market
    trader_weighted_sum = np.sum([np.sum(cluster) * np.sum(grid[cluster]) for cluster in clusters])

    # Normalising the sum
    log_return = trader_weighted_sum / beta

    return log_return


def convert_time(elapsed_time: int) -> str:
    """
    Convert time (in seconds) to time format X h Y m Z s.

    Args:
        elapsed_time: Time in seconds.

    Returns:
        String that contains the time in the new format.
    """
    # Calculating the hours
    hours, remainder = divmod(elapsed_time, 3600)

    # Calculating the minutes and seconds
    minutes, seconds = divmod(remainder, 60)

    return f"{hours} h {minutes} m {seconds} s"
