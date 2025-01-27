import logging
import time

import numpy as np
import matplotlib.pyplot as plt

from CAStockModel.model.base_model.utils.utility_elements import initialise_market_grid, find_clusters, calculate_log_return, convert_time
from CAStockModel.model.base_model.percolation_dyn import apply_percolation_dyn
from CAStockModel.model.base_model.stochastic_dyn import apply_stochastic_dynamics

from CAStockModel.model.trading_strat.utils.utility_elements import initialise_market_grid_for_coupling_strat
from CAStockModel.model.trading_strat.stochastic_dyn import stochastic_dynamics_coupling_strategy

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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


def run_coupled_stock_price_simulation(
        shape: tuple,
        active_ratio: float,
        p_e: float,
        p_d: float,
        p_h: float,
        A: float,
        a: float,
        h: float,
        beta: float,
        sensitivity_int: tuple,
        alpha: int,
        time_delay: int,
        sim_number: int=10
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the simulation of coupled stock market.

    Args:
        shape: Shape of the 2D grid consisting the traders. The multiplication of the length and the width will result in the number of traders on the market.
        active_ratio: Ratio of the active traders (buy or sell) in the initial grid.
        p_e: An inactive trader's probability of 'random' entering the market as active (not by the potential additional info gotten from active neighbours).
        p_d: Probability that an inactive trader will diffuse (make inactive) one of their active neighbours.
        p_h: Probability that an active trader will herd (make active) one of their inactive neighbours.
        A: Scaler of the general info connectivity in a cluster. (The higher the scaler is the more the traders in the cluster pay attention to the others in general without differentiation.)
        a: Scaler of the individual trader - trader connection in a cluster. The higher the scaler the more important the individual connection between the traders.
        h: Scaler of the external info relevancy for the clusters.
        beta: Normalization constant in the log return calculation
        sensitivity_int: The threshold that marks the point where the S1 normalized difference is deemed significant by the S2 trader, prompting a coupling strategy; otherwise, it is considered a random fluctuation.
        alpha: The length of time the S2 traders consider valuable and relevant in the underlying stock price S1 for S2.
        time_delay: The length of the time with which traders in smaller clusters get the information from the underlying market S1. This delay can be 0, t, 2*t, 3*t based on which quartile the cluster belongs based on its size.
        sim_number:

    Returns:
        #TODO

    """
    start_time = time.time()

    # Initialise grid for S1
    s1_initial_grid = initialise_market_grid(shape, active_ratio)

    # Initialise grid and individual trader sensibility for S2
    s2_initial_grid, trader_decision_sensitivity = initialise_market_grid_for_coupling_strat(shape,
                                                                                          active_ratio,
                                                                                          sensitivity_int)

    logging.info(f"Initialisation DONE   {convert_time(int(time.time() - start_time))}")

    # Create containers for simulation data
    s1_grid_history = [s1_initial_grid]
    s1_price_index = [100]
    s1_log_returns = []

    s2_grid_history = [s2_initial_grid]
    s2_price_index = [100]
    s2_log_returns = []

    coupling_strat_decision_tracker = []

    # Forward the underlying S1 price providing enough data for the S2 traders to make decisions on the two strategies
    for _ in range(alpha + 3 * time_delay):
        s1_grid_history.append(apply_percolation_dyn(s1_grid_history[-1], p_e, p_d, p_h))
        clusters = find_clusters(s1_grid_history[-1])
        s1_grid_history[-1] = apply_stochastic_dynamics(s1_grid_history[-1], clusters, A, a, h)

        s1_log_returns.append(calculate_log_return(s1_grid_history[-1], clusters, beta))
        s1_price_index.append(s1_price_index[-1] * np.exp(s1_log_returns[-1]))

    logging.info(f"Underlying Stock Price S1 preparation DONE   {convert_time(int(time.time() - start_time))}")

    for i in range(sim_number):
        # Calculate the next step in the underlying S1 market grid
        s1_grid_history.append(apply_percolation_dyn(s1_grid_history[-1], p_e, p_d, p_h))
        s1_clusters = find_clusters(s1_grid_history[-1])
        s1_grid_history[-1] = apply_stochastic_dynamics(s1_grid_history[-1], s1_clusters, A, a, h)

        # Calculate the next step in the underlying S1 stock price
        s1_log_returns.append(calculate_log_return(s1_grid_history[-1], s1_clusters, beta))
        s1_price_index.append(s1_price_index[-1] * np.exp(s1_log_returns[-1]))

        # Calculate the next step in the S2 market grid
        s2_grid_history.append(apply_percolation_dyn(s2_grid_history[-1], p_e, p_d, p_h))
        s2_clusters = find_clusters(s2_grid_history[-1])
        new_s2_grid, traders_ratio_following_coupling_strat = stochastic_dynamics_coupling_strategy(
            s2_grid_history[-1],
            trader_decision_sensitivity,
            s2_clusters,
            A,
            a,
            h,
            np.array(s1_price_index),
            time_delay,
            alpha)
        s2_grid_history[-1] = new_s2_grid
        coupling_strat_decision_tracker.append(traders_ratio_following_coupling_strat)

        # Calculate the next step in the S2 stock price
        s2_log_returns.append(calculate_log_return(s2_grid_history[-1], s2_clusters, beta))
        s2_price_index.append(s2_price_index[-1] * np.exp(s2_log_returns[-1]))

        if i % 50 == 0:
            logging.info(f"{i}th simulation DONE   {convert_time(int(time.time() - start_time))}")

    logging.info(f"Simulation DONE   {convert_time(int(time.time() - start_time))}")

    return (np.array(s1_grid_history)[alpha + 3 * time_delay:], np.array(s1_price_index)[alpha + 3 * time_delay:],
            np.array(s1_log_returns)[alpha + 3 * time_delay:], np.array(s2_grid_history), np.array(s2_price_index),
            np.array(s2_log_returns), np.array(coupling_strat_decision_tracker) * 100)


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
sensitivity_int = (1, 8)
alpha = 100
time_delay = 15

###################################################

# Plotting

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

(s1_grid, s1_price, s1_log_return, s2_grid,
 s2_price, s2_log_return, coupling_strat_tracker) = run_coupled_stock_price_simulation(shape,
                                                                                       active_ratio,
                                                                                       p_e,
                                                                                       p_d,
                                                                                       p_h,
                                                                                       A,
                                                                                       a,
                                                                                       h,
                                                                                       beta,
                                                                                       sensitivity_int,
                                                                                       alpha,
                                                                                       time_delay,
                                                                                       2000)
x_time = np.arange(len(s1_price))
s1_log_return = np.divide(s1_log_return - np.mean(s1_log_return), np.std(s1_log_return))
s2_log_return = np.divide(s2_log_return - np.mean(s2_log_return), np.std(s2_log_return))

ax[0, 0].plot(np.arange(len(s1_price)), s1_price, label='S1', color='blue')
ax[0, 0].plot(np.arange(len(s1_price)), s2_price, label='S2', color='red')
ax[0, 0].set_title('Price Index')
ax[0, 0].grid()
ax[0, 0].set_xlabel('Time')
ax[0, 0].set_ylabel('Price')
ax[0, 0].legend()

ax[0, 1].plot(np.arange(len(s1_log_return)), s1_log_return, label='S1', color='blue')
ax[0, 1].plot(np.arange(len(s1_log_return)), s2_log_return, label='S2', color='red')
ax[0, 1].set_title('Log Return')
ax[0, 1].grid()
ax[0, 1].set_xlabel('Time')
ax[0, 1].set_ylabel('Log Return')
ax[0, 1].legend()

ax[1, 0].plot(np.arange(len(coupling_strat_tracker)), coupling_strat_tracker)
ax[1, 0].set_title('Coupling Strat Usage Proportion')
ax[1, 0].grid()
ax[1, 0].set_xlabel('Time')
ax[1, 0].set_ylabel('Proportion')

plt.show()

log_return_correlation = np.corrcoef(s1_log_return, s2_log_return)
print(log_return_correlation)




#
# ###################################################
#
# # Plotting
#
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#
# for _ in range(3):
#     _, price_index, log_returns = run_stock_price_simulation(shape, active_ratio, p_e, p_d, p_h, A, a, h, beta, 10)
#     log_returns = np.divide(log_returns - np.mean(log_returns), np.std(log_returns))
#     ax[0].plot(np.arange(len(log_returns)), log_returns)
#     ax[1].plot(np.arange(len(price_index)), price_index)
#     print("check")
#
# ax[0].grid()
# ax[1].grid()
# ax[0].set_title('Log Return')
# ax[1].set_title('Price Index')
# ax[0].set_ylabel('Return')
# ax[1].set_ylabel('Price')
# ax[0].set_xlabel('Time')
# ax[1].set_xlabel('Time')
#
# plt.show()



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
