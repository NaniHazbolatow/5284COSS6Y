import time
from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

from utils.utility_elements import initialise_market_grid, find_clusters, calculate_log_return, convert_time
from model.percolation_dyn import apply_percolation_dyn
from model.stochastic_dyn import apply_stochastic_dynamics

np.random.seed(100)


def run_stock_price_simulation(
        shape: tuple,
        active_ratio: float,
        p_e: float,
        p_d: float,
        p_h: float,
        A: float,
        a: float,
        h: float,
        beta: float,
        sim_number: int=10
) -> [np.ndarray, np.ndarray, np.ndarray]:
    start_time = time.time()

    initial_grid = initialise_market_grid(shape, active_ratio)

    grid_history = [initial_grid]
    price_index = [100]
    log_returns = []

    for i in range(sim_number):
        grid_history.append(apply_percolation_dyn(grid_history[-1], p_e, p_d, p_h))

        clusters = find_clusters(grid_history[-1])

        grid_history[-1] = apply_stochastic_dynamics(grid_history[-1], clusters, A, a, h)

        log_return = calculate_log_return(grid_history[-1], clusters, beta)
        log_returns.append(log_return)

        price_index.append(price_index[-1] * np.exp(log_return))

        if i % 100 == 0:
            print(f"{i}th simulation DONE   {convert_time(int(time.time() - start_time))}")

    print(f"Simulation time is   {convert_time(int(time.time() - start_time))}")

    return np.array(grid_history), np.array(price_index), np.array(log_returns)


###################################################

# Settings

shape = (512, 128)
active_ratio = 17000 / (512 * 128)
p_e = 0.0001
p_d = 0.05
p_h = 0.0493
A = 1.6
a = 2 * A
h = 0
beta = shape[0] ** 2 * shape[1] ** 2

###################################################

# Plotting

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

for _ in range(3):
    _, price_index, log_returns = run_stock_price_simulation(shape, active_ratio, p_e, p_d, p_h, A, a, h, beta, 10)
    log_returns = np.divide(log_returns - np.mean(log_returns), np.std(log_returns))
    ax[0].plot(np.arange(len(log_returns)), log_returns)
    ax[1].plot(np.arange(len(price_index)), price_index)
    print("check")

ax[0].grid()
ax[1].grid()
ax[0].set_title('Log Return')
ax[1].set_title('Price Index')
ax[0].set_ylabel('Return')
ax[1].set_ylabel('Price')
ax[0].set_xlabel('Time')
ax[1].set_xlabel('Time')

plt.show()



# ###################################################
#
# # Settings
#
# shape = (512, 128)
# active_ratio = np.array([15000, 10000, 8000, 5000, 2500]) / (512 * 128)
# p_e = 0.0001
# p_d = 0.05
# p_h = [0.0493, 0.0490, 0.0488, 0.0485, 0.0475]
#
# ###################################################
#
# # Plotting
#
# active_trader = []
#
# for idx, p_h_value in enumerate(p_h):
#     seller, buyer, _ = run_stock_price_simulation(shape, active_ratio[idx], p_e, p_d, p_h_value)
#     active_trader.append(seller + buyer)
#
#
# fig = plt.figure(figsize=(10, 10))
#
# t = np.arange(len(active_trader[0]))
#
# for idx, ts_active_trader in enumerate(active_trader):
#     plt.plot(t, ts_active_trader, label=f"p_h = {p_h[idx]}")
#     print(f"check {idx} DONE")
#
# plt.title("Trader Type Proportion")
# plt.xlabel("Time")
# plt.ylabel("Proportion")
# plt.legend()
#
# plt.show()
