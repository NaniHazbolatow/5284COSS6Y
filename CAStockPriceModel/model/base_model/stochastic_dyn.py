import numpy as np
from numba import njit

def apply_stochastic_dynamics(grid: np.ndarray, clusters: np.ndarray, A: float, a: float, h: float) -> np.ndarray:
    """
    Applies the stochastic dynamics to existing grid: every trader who decided to be active in the next time step,
    decides, whether to buy or sell. This is based on an individually calculated probability of buying, which calculation
    is based on the stochastic internal dynamics (traders' trust and distrust towards each other and towards their
    cluster) of the clusters and the external force.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        clusters: 3D numpy array where every slice along the 0 axis consists of filter of traders belonging to the same clusters.
        A: General cluster level trust scaler (fixed model parameter)
        a: One to one trust scaler (fixed model parameter)
        h: External decision altering force scaler (fixed model parameter)

    Returns:
        Updated grid with the decisions of traders chosen to be active in the next time step.
    """
    # Initialising a new grid to avoid changing the input
    new_grid = grid.copy()

    # Iterating through the clusters as traders decisions are dependent on the traders belonging to the same cluster
    for current_cluster in clusters:
        # Extracting cluster specific info
        cluster_elements = grid[current_cluster]
        n = np.sum(current_cluster)

        # Calculating scaler and external force for every trader in the cluster
        A_ij, h_i = _simulate_connections(n, A, a, h)

        # Calculating the probability of buying for every trader in the cluster
        probabilities = _calculate_buying_probs(n, cluster_elements, A_ij, h_i)

        # Updating the grid with the decision made by the traders in the cluster
        new_grid[current_cluster] = _cluster_value_upgrade(probabilities)

    return new_grid


@njit
def _simulate_connections(n: int, A: float, a: float, h: float) -> [np.ndarray]:
    """
    Calculates the scalar and external force for the individual buying probability calculations.
    A_ij = A * random_uniform_var(-1, 1)_cluster + a * random_uniform_var(-1, 1)_ij
    h_i = h * random_uniform_var(-1, 1)_i

    Args:
        n: Number of traders in the cluster
        A: General cluster level trust scaler (fixed model parameter)
        a: One to one trust scaler (fixed model parameter)
        h: External decision altering force scaler (fixed model parameter)

    Returns:
        Scaler and external force for the individual buying probability calculations
    """
    # Calculating the internal elements of the scaler and external force formula
    xi_k = np.random.uniform(-1, 1)  # Cluster level trust
    nu_ij = np.random.uniform(-1, 1, size=(n, n))  # One to one trader level trust
    zeta_i = np.random.uniform(-1, 1, size=n)  # External force

    # Calculating the scaler and the external force
    A_ij = A * xi_k + a * nu_ij
    h_i = h * zeta_i

    return A_ij, h_i


@njit
def _calculate_buying_probs(n: int, cluster_elements: np.ndarray, A_ij: np.ndarray, h_i: np.ndarray) -> np.ndarray:
    """
    Calculates the probability of choosing to buy for every trader in the cluster.

    Args:
        n: Number of traders in the cluster.
        cluster_elements: previous decisions of the traders in the cluster.
        A_ij: Trader - trader connection trust scaler
        h_i: External (decision altering) force

    Returns:
        Traders' individual probabilities of buying
    """
    I_i = np.sum(A_ij * cluster_elements, axis=1) / n + h_i
    return np.divide(1, 1 + np.exp(-2 * I_i))


@njit
def _cluster_value_upgrade(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculates the decisions of the traders whether to buy or sell based on the individual probability.

    Args:
        probabilities: Individual probabilities of the traders of buying. (2D numpy array, same size as grid)

    Returns:
        Traders decisions represented with 1 or -1 values. (1D numpy array)
    """
    return np.where(np.random.random(len(probabilities)) > probabilities, -1, 1)
