import numpy as np

from CAStockModel.model.utils.utility_elements import initialise_market_grid
from CAStockModel.model.utils.constants import SENSITIVITY_LOWER_BOUND


def initialise_market_grid_for_coupling_strat(
        shape: tuple,
        active_ratio:  float,
        sensitivitiy_ub: float
) -> [np.ndarray, np.ndarray]:
    grid = initialise_market_grid(shape, active_ratio)
    trader_background_info = np.random.uniform(SENSITIVITY_LOWER_BOUND, sensitivitiy_ub, grid.shape)

    return grid, trader_background_info
