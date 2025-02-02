import logging
import time

import numpy as np

from CAStockModel.model.utils.utility_elements import (initialise_market_grid, find_clusters, calculate_log_return,
                                                       convert_time)
from CAStockModel.model.base_model.percolation_dyn import apply_percolation_dyn
from CAStockModel.model.base_model.stochastic_dyn import apply_stochastic_dynamics
from CAStockModel.model.analysis.signal_function import calc_singal_value, det_phase

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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
