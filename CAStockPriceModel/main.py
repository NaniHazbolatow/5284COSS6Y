import logging
import time

import numpy as np

from CAStockModel.model.utils.utility_elements import (initialise_market_grid, find_clusters, calculate_log_return,
                                                       convert_time)
from CAStockModel.model.base_model.percolation_dyn import apply_percolation_dyn
from CAStockModel.model.base_model.stochastic_dyn import apply_stochastic_dynamics
from CAStockModel.model.analysis.signal_function import calc_singal_value, det_phase
from CAStockModel.model.trading_strat.risk_thd import scale_risk_thd_by_previous_choice

from CAStockModel.model.trading_strat.utils.utility_elements import initialise_market_grid_for_coupling_strat
from CAStockModel.model.trading_strat.stochastic_dyn import stochastic_dynamics_coupling_strategy

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_coupled_stock_price_simulation(
        shape: tuple,
        active_ratio: float,
        starting_sens_ub: float,
        p_e: float,
        p_d: float,
        p_h: float,
        A: float,
        a: float,
        h: float,
        beta: float,
        alpha: int,
        time_delay: int,
        self_org: bool,
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
        For both markets (For S1 preparation time interval is excluded):
            Grid structure history (3D numpy array, 0 axis being time)
            Price Index History (1D numpy array, 0 axis being time)
            Log Return History (1D numpy array, 0 axis being time)
            Signal Value History (1D numpy array, 0 axis being time)
            Phase Index History (1D numpy array, 0 axis being time)
        For S2 only:
            Proportion History of traders choosing Coupling Strategy (1D numpy array, 0 axis being time)

    """
    start_time = time.time()

    # Initialise grid for S1
    s1_initial_grid = initialise_market_grid(shape, active_ratio)

    # Initialise grid and individual trader sensibility for S2
    s2_initial_grid, s2_trader_sensitivity = initialise_market_grid_for_coupling_strat(shape,
                                                                                             active_ratio,
                                                                                             starting_sens_ub)

    logging.info(f"Initialisation DONE   {convert_time(int(time.time() - start_time))}")

    # Create containers for simulation data
    s1_grid_history = [s1_initial_grid]
    s1_price_index = [100]
    s1_log_returns = []
    s1_signal_value_history = []
    s1_phase_history = []

    s2_grid_history = [s2_initial_grid]
    s2_price_index = [100]
    s2_log_returns = []
    s2_signal_value_history = []
    s2_phase_history = []

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

        # Calculate the signal value and the phase for S1
        s1_signal_value_history.append(calc_singal_value(s1_grid_history[-1], s1_clusters, 'quantile'))
        s1_phase_history.append(det_phase(s1_signal_value_history[-1]))

        # Calculate the next step in the S2 market grid
        s2_grid_history.append(apply_percolation_dyn(s2_grid_history[-1], p_e, p_d, p_h))
        s2_clusters = find_clusters(s2_grid_history[-1])
        new_s2_grid, traders_following_coupling_strat = stochastic_dynamics_coupling_strategy(
            s2_grid_history[-1],
            s2_trader_sensitivity,
            s2_clusters,
            A,
            a,
            h,
            np.array(s1_price_index),
            time_delay,
            alpha)
        s2_grid_history[-1] = new_s2_grid
        coupling_strat_decision_tracker.append(
            np.sum(traders_following_coupling_strat) / max([np.sum(np.isin(s2_grid_history[-1], [1, -1])), 1]))

        if self_org:
            # Recalculate the S2 trader sensitivities
            s2_trader_sensitivity = scale_risk_thd_by_previous_choice(
                s2_trader_sensitivity,
                s2_grid_history[-1],
                traders_following_coupling_strat)

        # Calculate the next step in the S2 stock price
        s2_log_returns.append(calculate_log_return(s2_grid_history[-1], s2_clusters, beta))
        s2_price_index.append(s2_price_index[-1] * np.exp(s2_log_returns[-1]))

        # Calculate the signal value and the phase for S2
        s2_signal_value_history.append(calc_singal_value(s2_grid_history[-1], s2_clusters, 'quantile'))
        s2_phase_history.append(det_phase(s2_signal_value_history[-1]))

        if i % 50 == 0:
            logging.info(f"{i}th simulation DONE   {convert_time(int(time.time() - start_time))}")

    logging.info(f"Simulation DONE   {convert_time(int(time.time() - start_time))}")

    return (np.array(s1_grid_history)[alpha + 3 * time_delay:], np.array(s1_price_index)[alpha + 3 * time_delay:],
            np.array(s1_log_returns)[alpha + 3 * time_delay:], np.array(s1_signal_value_history),
            np.array(s1_phase_history), np.array(s2_grid_history), np.array(s2_price_index),
            np.array(s2_log_returns), np.array(s2_signal_value_history), np.array(s2_phase_history),
            np.array(coupling_strat_decision_tracker) * 100)


# Not directly used in the main model, but for extra parameter anlysis
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
    signal_history = []
    phase_history = []

    logging.info(f"Initialisation DONE   {convert_time(int(time.time() - start_time))}")

    for i in range(sim_number):
        grid_history.append(apply_percolation_dyn(grid_history[-1], p_e, p_d, p_h))

        clusters = find_clusters(grid_history[-1])

        grid_history[-1] = apply_stochastic_dynamics(grid_history[-1], clusters, A, a, h)

        log_return = calculate_log_return(grid_history[-1], clusters, beta)
        log_returns.append(log_return)

        price_index.append(price_index[-1] * np.exp(log_return))

        signal_history.append(calc_singal_value(grid_history[-1], clusters, 'top'))
        phase_history.append(det_phase(signal_history[-1]))

        if i % 100 == 0:
            logging.info(f"{i}th simulation DONE   {convert_time(int(time.time() - start_time))}")

    logging.info(f"Simulation DONE   {convert_time(int(time.time() - start_time))}")

    return np.array(grid_history), np.array(price_index), np.array(log_returns), np.array(signal_history), np.array(phase_history)
