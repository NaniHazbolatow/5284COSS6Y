import numpy as np

from CAStockModel.model.utils.utility_elements import initialise_market_grid
from CAStockModel.model.utils.constants import SENSITIVITY_LOWER_BOUND


def initialise_market_grid_for_coupling_strat(
        shape: tuple,
        active_ratio:  float,
        sensitivitiy_ub: float
) -> [np.ndarray, np.ndarray]:
    """
    Initialises a grid (2D numpy array) representing the market and the individual risk appetite of the traders on the
    market, represented by a 2D numpy array same size as the grid, where the risk appetite index of a trader can be
    found at the corresponding index. Every cell in the former grid is going to take the number from the set {1, 0, -1}.
    The ratio of traders having 1 or -1 among all the traders is going to be equal to the predetermined value. The risk
    appetite index is sampled from a uniform distribution.

    Args:
        shape: Shape information of the market. (The multiplication of the two numbers results in the number of traders on the market.)
        active_ratio: Initial ratio of active traders (having either 1 or -1 value) among all the traders.
        sensitivitiy_ub: The upper bound of the uniform distribution, from which the risk appetite index is sampled.

    Returns:
        Grid representing the market and the individual risk appetite of the traders on the market.
    """
    # Initialise the market grid
    grid = initialise_market_grid(shape, active_ratio)

    # Sampling the risk appetite of the traders on the market
    trader_background_info = np.random.uniform(SENSITIVITY_LOWER_BOUND, sensitivitiy_ub, grid.shape)

    return grid, trader_background_info
