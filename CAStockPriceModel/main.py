import numpy as np
import matplotlib.pyplot as plt

from utils.utility_elements import initialise_market_grid
from model.percolation_dyn import apply_percolation_dyn


np.random.seed(100)


def run_stock_price_simulation(shape: tuple, active_ratio: float, p_e: float, p_d: float, p_h: float) -> np.array:
    initial_grid = initialise_market_grid(shape, active_ratio)

    grid_history = [initial_grid]

    for _ in range(4000):
        grid_history.append(apply_percolation_dyn(grid_history[-1], p_e, p_d, p_h))

    ts_for_seller = np.array([np.sum(grid == -1) for grid in grid_history])
    ts_for_buyer = np.array([np.sum(grid == 1) for grid in grid_history])
    ts_for_inactive = np.array([np.sum(grid == 0) for grid in grid_history])

    return ts_for_seller, ts_for_buyer, ts_for_inactive


###################################################

# Settings

shape = (512, 128)
active_ratio = np.array([15000, 10000, 8000, 5000, 2500]) / (512 * 128)
p_e = 0.0001
p_d = 0.05
p_h = [0.0493, 0.0490, 0.0488, 0.0485, 0.0475]

###################################################

# Plotting

active_trader = []

for idx, p_h_value in enumerate(p_h):
    seller, buyer, _ = run_stock_price_simulation(shape, active_ratio[idx], p_e, p_d, p_h_value)
    active_trader.append(seller + buyer)


fig = plt.figure(figsize=(10, 10))

t = np.arange(len(active_trader[0]))

for idx, ts_active_trader in enumerate(active_trader):
    plt.plot(t, ts_active_trader, label=f"p_h = {p_h[idx]}")
    print(f"check {idx} DONE")

plt.title("Trader Type Proportion")
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.legend()

plt.show()
