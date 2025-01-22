import numpy as np

def initialise_market_grid(shape: tuple, actice_ratio: float) -> np.ndarray:
    active_indices = np.random.choice(shape[0] * shape[1], replace=False, size=int(actice_ratio * shape[0] * shape[1]))

    one_d_grid = np.zeros(shape=shape[0] * shape[1], dtype=np.int8)
    one_d_grid[active_indices] = np.random.choice([1, -1], replace=True, size=len(active_indices))

    two_d_grid = one_d_grid.reshape(shape)

    return two_d_grid
