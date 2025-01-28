import numpy as np

from CAStockModel.model.utils.utility_elements import initialise_market_grid


def initialise_market_grid_for_coupling_strat(
        shape: tuple,
        active_ratio: float,
        price_diff_int: tuple
) -> [np.ndarray, np.ndarray]:
    grid = initialise_market_grid(shape, active_ratio)
    trader_background_info = np.random.uniform(price_diff_int[0], price_diff_int[1], grid.shape)

    return grid, trader_background_info
