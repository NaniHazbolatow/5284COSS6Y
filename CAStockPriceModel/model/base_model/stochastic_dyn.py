import numpy as np
from numba import njit

def apply_stochastic_dynamics(grid: np.ndarray, clusters: np.ndarray, A: float, a: float, h: float) -> np.ndarray:
    new_grid = grid.copy()

    for current_cluster in clusters:
        cluster_elements = grid[current_cluster]
        n = np.sum(current_cluster)

        A_ij, h_i = _simulate_connections(n, A, a, h)

        probabilities = _calculate_buying_probs(n, cluster_elements, A_ij, h_i)

        new_grid[current_cluster] = _cluster_value_upgrade(probabilities)

    return new_grid


@njit
def _simulate_connections(n: int, A: float, a: float, h: float) -> [np.ndarray]:
    xi_k = np.random.uniform(-1, 1)
    nu_ij = np.random.uniform(-1, 1, size=(n, n))
    zeta_i = np.random.uniform(-1, 1, size=n)

    A_ij = A * xi_k + a * nu_ij
    h_i = h * zeta_i

    return A_ij, h_i


@njit
def _calculate_buying_probs(n: int, cluster_elements: np.ndarray, A_ij: np.ndarray, h_i: np.ndarray) -> np.ndarray:
    I_i = np.sum(A_ij * cluster_elements, axis=1) / n + h_i
    return np.divide(1, 1 + np.exp(-2 * I_i))


@njit
def _cluster_value_upgrade(probabilities: np.ndarray) -> np.ndarray:
    return np.where(np.random.random(len(probabilities)) > probabilities, -1, 1)
