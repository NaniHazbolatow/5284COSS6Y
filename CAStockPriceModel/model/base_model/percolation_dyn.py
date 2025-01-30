import numpy as np
from numba import njit


def apply_percolation_dyn(grid: np.ndarray, p_e: float, p_d: float, p_h: float) -> np.ndarray:

    neighbours = _create_neighbours(grid)

    mask_1, mask_3, mask_4 = _create_masks(grid, neighbours)

    n_inactive_neighbour, n_active_neighbour = _count_neighbours(neighbours)
    prob_1, prob_3, prob_4 = _calculate_probabilities(n_inactive_neighbour, n_active_neighbour, p_e, p_d, p_h)

    eff_filter_1, eff_filter_3, eff_filter_4 = _create_effective_filters(grid,
                                                                         prob_1,
                                                                         prob_3,
                                                                         prob_4,
                                                                         mask_1,
                                                                         mask_3,
                                                                         mask_4)
    new_grid = _create_new_grid(grid, eff_filter_1, eff_filter_3, eff_filter_4)

    return new_grid


def _create_neighbours(grid: np.ndarray) -> np.ndarray:
    padded_grid = np.pad(grid, pad_width=1, mode="constant", constant_values=0)

    top_neighbours = padded_grid[:-2, 1:-1]
    bottom_neighbours = padded_grid[2:, 1:-1]
    left_neighbours = padded_grid[1:-1, :-2]
    right_neighbours = padded_grid[1:-1, 2:]

    return np.stack([top_neighbours, bottom_neighbours, left_neighbours, right_neighbours], axis=0)


def _create_masks(grid: np.ndarray, neighbours: np.ndarray) -> [np.ndarray]:
    # Type 1: Inactive cell with Inactive neighbours
    mask_1 = (grid == 0) & np.all(neighbours == 0, axis=0)

    # Type 3: Inactive cell with AT LEAST one Active neighbour
    mask_3 = (grid == 0) & np.any(np.isin(neighbours, [1, -1]), axis=0)

    # Type 4: Active cell with AT LEAST one Inactive neighbour
    mask_4 = (np.isin(grid, [1, -1])) & np.any(neighbours == 0, axis=0)

    return mask_1, mask_3, mask_4


def _count_neighbours(neighbours: np.ndarray) -> [np.ndarray]:
    n_inactive_neighbour = np.sum(neighbours == 0, axis=0)
    n_active_neighbour = np.sum(np.isin(neighbours, [1, -1]), axis=0)

    return n_inactive_neighbour, n_active_neighbour


@njit
def _calculate_probabilities(
        inactive_neighbour: np.ndarray,
        active_neighbour: np.ndarray,
        p_e: float,
        p_d: float,
        p_h: float
) -> [float]:
    prob_1 = p_e
    prob_3 = 1 - np.power((1 - p_h), active_neighbour) * (1 - p_e)
    prob_4 = 1 - np.power((1 - p_d), inactive_neighbour)

    return prob_1, prob_3, prob_4


@njit
def _create_effective_filters(
        grid: np.ndarray,
        prob_1: float,
        prob_3: float,
        prob_4: float,
        mask_1: np.ndarray,
        mask_3: np.ndarray,
        mask_4: np.ndarray
) -> [np.ndarray]:
    prob_grid = np.random.random(size=(grid.shape[0], grid.shape[1]))

    type_1_filter = (prob_grid < prob_1) & mask_1
    type_3_filter = (prob_grid < prob_3) & mask_3
    type_4_filter = (prob_grid < prob_4) & mask_4

    return type_1_filter, type_3_filter, type_4_filter


def _create_new_grid(
        grid: np.ndarray,
        eff_filter_1: np.ndarray,
        eff_filter_3: np.ndarray,
        eff_filter_4: np.ndarray
) -> np.ndarray:
    new_grid = grid.copy()

    new_grid[eff_filter_1] = np.random.choice([1, -1], size=np.sum(eff_filter_1), replace=True)
    new_grid[eff_filter_3] = np.random.choice([1, -1], size=np.sum(eff_filter_3), replace=True)
    new_grid[eff_filter_4] = 0

    return new_grid
