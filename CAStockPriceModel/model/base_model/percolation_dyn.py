import numpy as np
from numba import njit


def apply_percolation_dyn(grid: np.ndarray, p_e: float, p_d: float, p_h: float) -> np.ndarray:
    """
    Applies the percolation dynamics to an existing grid. Every trader is going to decide to be active (having value of
    1 or -1) or inactive (having value of 0) in the next time step. There is a predefined probability of an active
    member turns their von Neumann inactive neighbour to active, and vica versa. The probabilities are independent.
    There is an additional chance of randomly entering the market. This does not depend on the trader's neighbour.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        p_e: Probability of randomly entering the market as inactive trader (being active in the next time step).
        p_d: Probability of an inactive trader turns one of their von Neumann active neighbour inactive.
        p_h: Probability of an active trader turns one of their von Neumann inactive neighbour to active.

    Returns:
        Updated Grid (2D numpy array)
    """
    # Creating the neighbour arrays
    neighbours = _create_neighbours(grid)

    # Creating filters for the traders in the possible 4 neighbour structure
    mask_1, mask_3, mask_4 = _create_masks(grid, neighbours)

    # Calculating the probabilities of the traders of changing their previous decision
    n_inactive_neighbour, n_active_neighbour = _count_neighbours(neighbours)
    prob_1, prob_3, prob_4 = _calculate_probabilities(n_inactive_neighbour, n_active_neighbour, p_e, p_d, p_h)

    # Calculating the new decisions and updating the previous grid
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
    """
    Creates an array which contains the neighbours of each trader in the grid. The array is 3D numpy array, where the
    indices along the 0 axis represent the top, bottom, left, right neighbour structure. A trader's neighbour is going
    to be on the same index on the corresponding 2D numpy array.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.

    Returns:
        3D numpy array consisting of the neighbours.
    """
    # Wrapping an inactive trader belt around the existing grid, representing that traders on the clusters' edge can
    # turn only inactive
    padded_grid = np.pad(grid, pad_width=1, mode="constant", constant_values=0)

    # Locating the von Neumann neighbours of the different directions
    top_neighbours = padded_grid[:-2, 1:-1]
    bottom_neighbours = padded_grid[2:, 1:-1]
    left_neighbours = padded_grid[1:-1, :-2]
    right_neighbours = padded_grid[1:-1, 2:]

    return np.stack([top_neighbours, bottom_neighbours, left_neighbours, right_neighbours], axis=0)


def _create_masks(grid: np.ndarray, neighbours: np.ndarray) -> [np.ndarray]:
    """
    Creates filters for the traders in the four possible neighbours structures.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        neighbours: 3D array consisting different directional neighbours on each slice along the 0 axis

    Returns:
        Trader filters (2D numpy array with boolean values)
    """
    # Type 1: Inactive cell with Inactive neighbours
    mask_1 = (grid == 0) & np.all(neighbours == 0, axis=0)

    # Type 3: Inactive cell with AT LEAST one Active neighbour
    mask_3 = (grid == 0) & np.any(np.isin(neighbours, [1, -1]), axis=0)

    # Type 4: Active cell with AT LEAST one Inactive neighbour
    mask_4 = (np.isin(grid, [1, -1])) & np.any(neighbours == 0, axis=0)

    return mask_1, mask_3, mask_4


def _count_neighbours(neighbours: np.ndarray) -> [np.ndarray]:
    """
    Calculates the number of active and inactive neighbours for every trader.

    Args:
        neighbours: 3D array consisting different directional neighbours on each slice along the 0 axis

    Returns:
        Number of active and inactive neighbours for every trader (2D numpy array same size as the grid) where every
        index contains the corresponding traders neighbour info
    """
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
    """
    Calculates the probability of different decisions a trader can make.

    Args:
        inactive_neighbour: 2D array consisting of the inactive neighbour info for every trader
        active_neighbour: 2D array consisting of the active neighbour info for every trader
        p_e: Probability of randomly entering the market as inactive trader (being active in the next time step).
        p_d: Probability of an inactive trader turns one of their von Neumann active neighbour inactive.
        p_h: Probability of an active trader turns one of their von Neumann inactive neighbour to active.

    Returns:
        Probabilities of the different possible events
    """
    prob_1 = p_e  # Probability of randomly entering a market
    prob_3 = 1 - np.power((1 - p_h), active_neighbour) * (1 - p_e)  # Probability of becoming active due to active neighbours
    prob_4 = 1 - np.power((1 - p_d), inactive_neighbour)  # Probability of turning inactive due to inactive neighbours

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
    """
    Calculating the decisions of every trader.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        prob_1: Probability of a trader randomly entering a market
        prob_3: Probability of a trader becoming active due to active neighbours
        prob_4: Probability of a trader turning inactive due to inactive neighbours
        mask_1: Filter for traders who are inactive and has no active neighbour
        mask_3: Filter for traders who are inactive and has at least one active neighbour
        mask_4: Filter for traders who are active and has at least one inactive neighbour

    Returns:
        Decisions of every trader (2D numpy array same size as the grid) Every filter represents a certain decision:
        filter_1: Traders, who are inactive with no active neighbours, turn active
        filter_3: Traders, who are inactive with at least one active neighbour, turn active
        filter_4: Traders, who are active and has at least one inactive neighbour, turn inactive
    """
    # Sampling random numbers for every trader from the range of [0, 1]
    prob_grid = np.random.random(size=(grid.shape[0], grid.shape[1]))

    # Calculating the decisions
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
    """
    Updates the current grid with the traders decisions. If an inactive trader turns active, their decision is randomly
    chosen from the set {1, -1}.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        eff_filter_1: Traders, who are inactive with no active neighbours, turn active
        eff_filter_3: Traders, who are inactive with at least one active neighbour, turn active
        eff_filter_4: Traders, who are active and has at least one inactive neighbour, turn inactive

    Returns:
        Updated grid (2D numpy array consisting of the decisions of the traders.
    """
    # Creating a new grid, avoiding the change of the input
    new_grid = grid.copy()

    # Applying the traders' decisions
    new_grid[eff_filter_1] = np.random.choice([1, -1], size=np.sum(eff_filter_1), replace=True)
    new_grid[eff_filter_3] = np.random.choice([1, -1], size=np.sum(eff_filter_3), replace=True)
    new_grid[eff_filter_4] = 0

    return new_grid
