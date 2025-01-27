import time
from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

from utils.utility_elements import (
    initialise_market_grid,
    find_clusters,
    calculate_log_return,
    convert_time,
)
from model.percolation_dyn import (
    apply_percolation_dyn,
    apply_percolation_dyn_heterogenous,
)
from model.stochastic_dyn import apply_stochastic_dynamics

from model.abm import generate_correlated_values, compute_moving_average_and_volatility, construct_signal

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
    sim_number: int = 10,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    start_time = time.time()

    initial_grid = initialise_market_grid(shape, active_ratio)

    grid_history = [initial_grid]
    price_index = [100]
    log_returns = []

    for i in range(sim_number):
        grid_history.append(apply_percolation_dyn(grid_history[-1], p_e, p_d, p_h))

        clusters = find_clusters(grid_history[-1])

        grid_history[-1] = apply_stochastic_dynamics(
            grid_history[-1], clusters, A, a, h
        )

        log_return = calculate_log_return(grid_history[-1], clusters, beta)
        log_returns.append(log_return)

        price_index.append(price_index[-1] * np.exp(log_return))

        if i % 100 == 0:
            print(
                f"{i}th simulation DONE   {convert_time(int(time.time() - start_time))}"
            )

    print(f"Simulation time is   {convert_time(int(time.time() - start_time))}")

    return np.array(grid_history), np.array(price_index), np.array(log_returns)


def run_stock_price_simulation_2(
    shape: tuple,
    active_ratio: float,
    p_e: float,
    p_d: float,
    p_h: float,
    A: float,
    a: float,
    h: float,
    beta: float,
    sim_number: int = 10,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    start_time = time.time()

    initial_grid_1 = initialise_market_grid(shape, active_ratio)
    initial_grid_2 = initialise_market_grid(shape, 0)

    grid_history_1 = [initial_grid_1]
    grid_history_2 = [initial_grid_2]
    price_index_1 = [100]
    price_index_2 = [100]
    log_returns_1 = []
    log_returns_2 = []

    risk_apt, look_back = generate_correlated_values([1, 5], [10, 50], 0.7, shape)
    for i in range(sim_number):
        # Run Simulation 1: Complete Homogenous
        grid_history_1.append(apply_percolation_dyn(grid_history_1[-1], p_e, p_d, p_h))

        clusters = find_clusters(grid_history_1[-1])

        grid_history_1[-1] = apply_stochastic_dynamics(
            grid_history_1[-1], clusters, A, a, h
        )

        log_return_1 = calculate_log_return(grid_history_1[-1], clusters, beta)
        log_returns_1.append(log_return_1)

        price_index_1.append(price_index_1[-1] * np.exp(log_return_1))

        # Now input price data into simulation 2: Complete Heterogenous
        ma, vol = compute_moving_average_and_volatility(
            np.array(price_index_1), look_back
        )
        print(ma)
        print(vol)

        signal = construct_signal(np.array(price_index_1), ma, vol, risk_apt)


        # We can make this probabilistic by:
        #probability_matrix = np.random.rand(shape[0], shape[1])  # Random probabilities between 0 and 1

        #p_e_2 = 0.05
        #decision_matrix = np.full((shape[0], shape[1]), p_e_2)

        #action_matrix = (decision_matrix > probability_matrix).astype(int) 

        grid_history_2.append(
            apply_percolation_dyn_heterogenous(grid_history_2[-1], p_d, p_h, signal)
        )

        clusters = find_clusters(grid_history_2[-1])

        grid_history_2[-1] = apply_stochastic_dynamics(
            grid_history_2[-1], clusters, A, a, h
        )

        log_return_2 = calculate_log_return(grid_history_2[-1], clusters, beta)
        log_returns_2.append(log_return_2)

        price_index_2.append(price_index_2[-1] * np.exp(log_return_2))

        if i % 100 == 0:
            print(
                f"{i}th simulation DONE   {convert_time(int(time.time() - start_time))}"
            )

    print(f"Simulation time is   {convert_time(int(time.time() - start_time))}")

    return  np.array(grid_history_1),  np.array(price_index_1),  np.array(log_returns_1), np.array(grid_history_2), np.array(price_index_2), np.array(log_returns_2),
    


###################################################

# Settings

shape = (20, 20)
#active_ratio = 17000 / (512 * 128)
active_rate = 0.2
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
    _,_,price_index_1,_, price_index_2, log_returns = run_stock_price_simulation_2(
        shape, active_rate, p_e, p_d, p_h, A, a, h, beta, 100
    )
    log_returns = np.divide(log_returns - np.mean(log_returns), np.std(log_returns))
    ax[0].plot(np.arange(len(price_index_1)), price_index_1)
    ax[1].plot(np.arange(len(price_index_2)), price_index_2 )
    print("check")

ax[0].grid()
ax[1].grid()
ax[0].set_title("Log Return")
ax[1].set_title("Price Index")
ax[0].set_ylabel("Return")
ax[1].set_ylabel("Price")
ax[0].set_xlabel("Time")
ax[1].set_xlabel("Time")

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


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
shape = (50, 50)
#active_ratio = 17000 / (512 * 128)
active_rate = 0.2
p_e = 0.0001
p_d = 0.05
p_h = 0.0493
A = 1.6
a = 2 * A
h = 0
beta = shape[0] ** 2 * shape[1] ** 2
sim_number = 1000


grid_history_1, price_index_1, _, grid_history_2, price_index_2, _ = run_stock_price_simulation_2(
    shape, active_rate, p_e, p_d, p_h, A, a, h, beta, sim_number
)


# Create the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
vmin, vmax = -1, 1  # Value range for both grids

# Initialize the plots
image1 = axs[0, 0].imshow(grid_history_1[0], cmap="coolwarm", animated=True,  vmin=vmin, vmax=vmax)
axs[0, 0].set_title("Simulation 1: Grid")

image2 = axs[0, 1].imshow(grid_history_2[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
axs[0, 1].set_title("Simulation 2: Grid")

line1, = axs[1, 0].plot([], [], color="blue")
axs[1, 0].set_xlim(0, sim_number)
axs[1, 0].set_ylim(min(price_index_1.min(), price_index_1.min()), max(price_index_1.max(), price_index_1.max()))
axs[1, 0].set_title("Simulation 1: Price Index")
axs[1, 0].set_xlabel("Simulation Step")
axs[1, 0].set_ylabel("Price Index")

line2, = axs[1, 1].plot([], [], color="green")
axs[1, 1].set_xlim(0, sim_number)
axs[1, 1].set_ylim(min(price_index_2.min(), price_index_2.min()), max(price_index_2.max(), price_index_2.max()))
axs[1, 1].set_title("Simulation 2: Price Index")
axs[1, 1].set_xlabel("Simulation Step")
axs[1, 1].set_ylabel("Price Index")

# Update function for animation
def update(frame):
    image1.set_array(grid_history_1[frame])
    image2.set_array(grid_history_2[frame])

    line1.set_data(range(frame + 1), price_index_1[:frame + 1])
    line2.set_data(range(frame + 1), price_index_2[:frame + 1])

    return image1, image2, line1, line2

# Create the animation
ani = FuncAnimation(fig, update, frames=sim_number, interval=100, blit=True)

# Display the animation
plt.tight_layout()
plt.show()