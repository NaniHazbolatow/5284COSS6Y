#utility_elements.py

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


def calculate_log_return(grid: np.ndarray, clusters: np.ndarray, beta: float) -> float:

    trader_weighted_sum = np.sum([np.sum(cluster) * np.sum(grid[cluster]) for cluster in clusters])
    log_return = trader_weighted_sum / beta

    return log_return


def convert_time(elapsed_time: int) -> str:
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours} h {minutes} m {seconds}"

def scale_probabilities(current_shape: tuple, p_e: float, p_d: float, p_h: float) -> tuple:
    """
    Scale probability parameters based on market size relative to reference shape (512, 128).
    
    Parameters:
    current_shape (tuple): Current market shape (height, width)
    p_e (float): Base spontaneous activation probability
    p_d (float): Base deactivation probability
    p_h (float): Base activation probability due to neighbors
    
    Returns:
    tuple: Scaled probabilities (p_e_scaled, p_d_scaled, p_h_scaled)
    """
    # Reference shape
    ref_height, ref_width = 512, 128
    ref_size = ref_height * ref_width
    
    # Current size
    current_size = current_shape[0] * current_shape[1]
    
    # Calculate scaling factor
    scaling_factor = current_size / ref_size
    
    # Scale probabilities
    # For larger markets, probabilities should decrease, and vice versa
    p_e_scaled = p_e / scaling_factor
    p_d_scaled = p_d / scaling_factor
    p_h_scaled = p_h / scaling_factor
    
    return p_e_scaled, p_d_scaled, p_h_scaled