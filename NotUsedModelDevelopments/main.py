from CAStockModel.model.utils.utility_elements import (initialise_market_grid, find_clusters, calculate_log_return,
                                                       convert_time)
from CAStockModel.model.base_model.percolation_dyn import apply_percolation_dyn
from CAStockModel.model.base_model.stochastic_dyn import apply_stochastic_dynamics
from CAStockModel.model.analysis.signal_function import calc_singal_value, det_phase

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import logging
import time
import os
from scipy.ndimage import label
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec  # For custom layout
from scipy.stats import norm
import yfinance as yf
import scipy.stats as stats

from CAStockModel.model.utils.constants import figure_dir
from CAStockModel.model.trading_strat.risk_thd import scale_risk_thd_by_previous_choice

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
) -> np.ndarray:
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

# Michelangelo part
def power_law(x, a, b):
    return a * x ** (-b)

def find_clusters2(grid): 
    """Identify clusters using von Neumann neighborhood."""
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])
    active = (grid != 0)
    clusters, n_clusters = label(active, structure=structure)
    
    # Get cluster sizes (excluding background)
    cluster_sizes = [np.sum(clusters == i) for i in range(1, n_clusters + 1)]
    
    return cluster_sizes  # Return list of cluster sizes

def check_plot_6(grids, x_limits=None, y_limits=None):
    """
    Computes and plots the cluster size distribution across multiple grids.
    
    Parameters:
        grids (list of 2D arrays): List of 2D grid arrays.
        x_limits (tuple, optional): Limits for x-axis.
        y_limits (tuple, optional): Limits for y-axis.
    """
    all_cluster_sizes = []

    # Process each grid to extract cluster sizes
    for grid in grids:
        cluster_sizes = find_clusters2(grid)
        all_cluster_sizes.extend(cluster_sizes)  # Collect all cluster sizes

    # Convert to numpy array
    all_cluster_sizes = np.array(all_cluster_sizes)

    # Compute unique sizes and their probabilities
    sizes, counts = np.unique(all_cluster_sizes, return_counts=True)
    normalized_counts = counts / np.sum(counts)  # Normalize to get probabilities

    # Fit the data to a power-law
    popt, _ = curve_fit(power_law, sizes, normalized_counts, p0=[1, 2])
    a, lambda_exponent = popt

    # Plot results
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(sizes, normalized_counts, color="black", s=10, label="Data") 
    x_fit = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
    plt.plot(x_fit, power_law(x_fit, *popt), linestyle="--", color="black", 
             label=f"$\\lambda \\sim {lambda_exponent:.1f}$, ph = {p_h}")  # Fitted line
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("S (Cluster Size)")
    plt.ylabel(r"$\rho$ (Probability)")
    plt.title("Cluster Size Distribution Across Grids")
    plt.legend()

    # Apply axis limits if specified
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)

    # Improve plot aesthetics
    axxx = plt.gca()
    axxx.spines['top'].set_visible(False)
    axxx.spines['right'].set_visible(False)
    plt.show()


shape = (256, 256)
active_ratio = 17000 / (512 * 128)
p_e = 0.0001
p_d = 0.05
A = 1.8
a = 2 * A
h = 0
beta = shape[0] ** 2 * shape[1] ** 2
p_h_vals = [ 0.0493, 0.0475]
for p_h in p_h_vals:
    grid_hist, price, log_return, signal_hist, phase_hist = run_stock_price_simulation(
        shape,
        active_ratio,
        p_e,
        p_d,
        p_h,
        A,
        a,
        h,
        beta,
        2000)
    
    check_plot_6(grid_hist, x_limits=(1, 100), y_limits=(1e-4, 1e-1))

fig, ax = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)


def check_plot_2(array, time_steps):
    array = np.array(array)
    array = array.flatten()  # check array is 1D
    normalized_returns = (array - np.mean(array)) / np.std(array)
    ax[0, 0].plot(time_steps, normalized_returns, color="purple", linestyle="-", linewidth=2) 
    ax[0, 0].set_xlabel("Time steps")
    ax[0, 0].set_ylabel("Normalized logarithmic returns")
    ax[0, 0].set_title("Time series of returns from model")

def check_plot_3(array):
    array = np.array(array)
    normalized_returns = (array - np.mean(array)) / np.std(array)  # Normalize returns

    # Create histogram of normalized returns
    bins = np.linspace(-10, 10, 100)  
    hist, edges = np.histogram(normalized_returns, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2  

    # Gaussian distribution for comparison
    x = np.linspace(-5, 5, 500)
    gaussian_pdf = norm.pdf(x, loc=0, scale=1)
    colors = ["red", "blue", "green", "purple", "orange", "black", "yellow", "cyan", "magenta", "pink"]
    random_color = np.random.choice(colors)

    ax[0, 1].plot(x, gaussian_pdf, linestyle='--', color="blue", label="Gaussian distribution")
    ax[0, 1].plot(bin_centers, hist, 'o', color = random_color, label="Model (Normalized Returns)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylim(1e-4, None)
    ax[0, 1].set_xlabel("r")
    ax[0, 1].set_ylabel(r"$\rho$")
    ax[0, 1].set_title("Probability Distribution of Returns")
    ax[0, 1].legend(loc='upper left')


def check_plot_3_5(array):
    array = np.array(array)
    normalized_returns = (array - np.mean(array)) / np.std(array)  # Normalize returns

    # Create histogram of normalized returns
    bins = np.linspace(-10, 10, 100)  
    hist, edges = np.histogram(normalized_returns, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2  

    # Gaussian distribution for comparison
    x = np.linspace(-5, 5, 500)
    gaussian_pdf = norm.pdf(x, loc=0, scale=1)
    colors = ["red", "blue", "green", "purple", "orange", "black", "yellow", "cyan", "magenta", "pink"]
    random_color = np.random.choice(colors)

    ax[0, 1].plot(x, gaussian_pdf, linestyle='--', color="blue", label="Gaussian distribution")
    ax[0, 1].plot(bin_centers, hist, 'o', color = random_color, label="AEX (Normalized Returns)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylim(1e-4, None)
    ax[0, 1].set_xlabel("r")
    ax[0, 1].set_ylabel(r"$\rho$")
    ax[0, 1].set_title("Probability Distribution of Returns")
    ax[0, 1].legend(loc='upper left')

def check_plot_4(array):
    array = np.array(array)
    normalized_returns = (array - np.mean(array)) / np.std(array)
    # Create Q-Q plot
    stats.probplot(normalized_returns, dist="norm", plot=ax[1, 1])

    colors = ["red", "blue", "green", "purple", "orange", "black", "yellow", "cyan", "magenta", "pink"]
    random_color = np.random.choice(colors)

    for line in ax[1, 1].get_lines():
        line.set_color(random_color)  
        line.set_markerfacecolor(random_color)  
        line.set_markeredgecolor(random_color)  


    # Labels and title
    ax[1, 1].set_title("Q-Q Plot of Log Returns vs. Normal Distribution")
    ax[1, 1].set_xlabel("Theoretical Quantiles")
    ax[1, 1].set_ylabel("Sample Quantiles")    

def check_plot_5(array, time_steps):
    array = np.array(array)
    array = array.flatten()  # check array is 1D
    normalized_returns = (array - np.mean(array)) / np.std(array)
    ax[1, 0].plot(time_steps, normalized_returns, color="purple", linestyle="-", linewidth=2) 
    ax[1, 0].set_xlabel("Time steps")
    ax[1, 0].set_ylabel("Normalized logarithmic returns")
    ax[1, 0].set_title("Time series of returns AEX")

# end functions Michelangelo

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
) -> np.ndarray:
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

#Continuation Michelangelo part
    time_steps = [i for i in range(len(s1_log_returns))]
    check_plot_2(s1_log_returns, time_steps)


    # Fetch historical market data
    ticker_symbol = "^AEX"
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(start="2005-01-01")
    days_counted = 5000
    data_sliced = historical_data[:days_counted]
    closed_prices = data_sliced["Close"]
    logR_AEX = np.diff(np.log(closed_prices))
    time_steps_AEX = [i for i in range(len(logR_AEX))]

    check_plot_3(s1_log_returns)
    check_plot_3_5(logR_AEX)

    check_plot_4(logR_AEX)
    check_plot_4(s1_log_returns)

    check_plot_5(logR_AEX, time_steps_AEX)

    plt.tight_layout()
    plt.show()
    
# end Michelangelo part
    return (np.array(s1_grid_history)[alpha + 3 * time_delay:], np.array(s1_price_index)[alpha + 3 * time_delay:],
            np.array(s1_log_returns)[alpha + 3 * time_delay:], np.array(s1_signal_value_history),
            np.array(s1_phase_history), np.array(s2_grid_history), np.array(s2_price_index),
            np.array(s2_log_returns), np.array(s2_signal_value_history), np.array(s2_phase_history),
            np.array(coupling_strat_decision_tracker) * 100)


if __name__ == '__main__':
    ################################################

    # Settings

    shape = (256, 256)
    active_ratio = 0.25
    initial_sens_ub = 5
    p_e = 0.0001
    p_d = 0.05
    p_h = 0.0493
    A = 1.8
    a = 2 * A
    h = 0
    beta = shape[0] ** 2 * shape[1] ** 2
    alpha = 100
    time_delay = 25
    self_organization = True

    ################################################

    # Animation

    (s1_grid, s1_price, s1_log_ret, s1_signal, s1_phase,
     s2_grid, s2_price, s2_log_ret, s2_signal, s2_phase, coupling_strat_tracker) = run_coupled_stock_price_simulation(
        shape,
        active_ratio,
        initial_sens_ub,
        p_e,
        p_d,
        p_h,
        A,
        a,
        h,
        beta,
        alpha,
        time_delay,
        self_organization,
        1000
    )

    # Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    vmin, vmax = -1, 1  # Value range for both grids

    # Initialize the plots for the two grids
    image1 = axs[0].imshow(s1_grid[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
    axs[0].set_title("Simulation 1: Grid")

    image2 = axs[1].imshow(s2_grid[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
    axs[1].set_title("Simulation 2: Grid")


    # Update function for animation
    def update(frame):
        # Update the grid lattice for both simulations
        image1.set_array(s1_grid[frame])
        image2.set_array(s2_grid[frame])

        return image1, image2


    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(s1_grid), interval=100, blit=True)

    # Save the animation as HTML
    ani.save(os.path.join(figure_dir, 'grid_animation_11.html'), writer='html', fps=5)

    # Display the animation
    plt.tight_layout()
    plt.show()




    ###################################################

    # Plotting

    # (s1_grid, s1_price, s1_log_ret, s1_signal, s1_phase,
    #  s2_grid, s2_price, s2_log_ret, s2_signal, s2_phase, coupling_strat_tracker) = run_coupled_stock_price_simulation(
    #     shape,
    #     active_ratio,
    #     p_e,
    #     p_d,
    #     p_h,
    #     A,
    #     a,
    #     h,
    #     beta,
    #     sensitivity_int,
    #     alpha,
    #     time_delay,
    #     500
    # )

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # Top-left subplot (Price Index)
    ax[0, 0].plot(s1_price, label="S1", color="blue")
    ax[0, 0].set_title('Price Index')
    ax[0, 0].set_ylabel('Price (S1)', color='blue')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].grid()
    ax[0, 0].tick_params(axis='y', labelcolor='blue')

    ax_alt_1 = ax[0, 0].twinx()
    ax_alt_1.plot(s2_price, label="S2", color="red")
    ax_alt_1.set_ylabel('Price (S2)', color='red')
    ax_alt_1.tick_params(axis='y', labelcolor='red')

    # Top-right subplot (Log Return)
    ax[0, 1].plot(s1_log_ret, label="S1", color="blue")
    ax[0, 1].set_title('Log Return')
    ax[0, 1].set_ylabel('Log Return (S1)', color='blue')
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].grid()
    ax[0, 1].tick_params(axis='y', labelcolor='blue')

    ax_alt_2 = ax[0, 1].twinx()
    ax_alt_2.plot(s2_log_ret, label="S2", color="red")
    ax_alt_2.set_ylabel('Log Return (S2)', color='red')
    ax_alt_2.tick_params(axis='y', labelcolor='red')

    # Bottom-left subplot (Phase Values)
    ax[1, 0].plot(s1_phase, label="S1", color="blue")
    ax[1, 0].plot(s2_phase, label="S2", color="red")
    ax[1, 0].set_title('Phase Values')
    ax[1, 0].set_ylabel('Phase Index (S1)', color='blue')
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].grid()
    ax[1, 0].tick_params(axis='y', labelcolor='blue')

    # Bottom-right subplot (Signal Function)
    ax[1, 1].plot(s1_signal, label="S1", color="blue")
    ax[1, 1].set_title('Signal Function')
    ax[1, 1].set_ylabel('Signal Function (S1)', color='blue')
    ax[1, 1].set_xlabel('Time')
    ax[1, 1].grid()
    ax[1, 1].tick_params(axis='y', labelcolor='blue')

    ax_alt_4 = ax[1, 1].twinx()
    ax_alt_4.plot(s2_signal, label="S2", color="red")
    ax_alt_4.set_ylabel('Signal Function (S2)', color='red')
    ax_alt_4.tick_params(axis='y', labelcolor='red')

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'test_11_plot.png'), dpi=500, bbox_inches='tight', format='png')
    plt.show()




#
# ###################################################
#
# # Settings
#
# shape = (512, 128)
# active_ratio = 17000 / (512 * 128)
# p_e = 0.0001
# p_d = 0.05
# p_h = 0.0493
# A = 1.6
# a = 2 * A
# h = 0
# beta = shape[0] ** 2 * shape[1] ** 2
# sensitivity_int = (1, 8)
# alpha = 100
# time_delay = 15
#
# ###################################################
#
# # Plotting
#
# fig, ax = plt.subplots(2, 2, figsize=(12, 12))
#
# (s1_grid, s1_price, s1_log_return, s2_grid,
#  s2_price, s2_log_return, coupling_strat_tracker) = run_coupled_stock_price_simulation(shape,
#                                                                                        active_ratio,
#                                                                                        p_e,
#                                                                                        p_d,
#                                                                                        p_h,
#                                                                                        A,
#                                                                                        a,
#                                                                                        h,
#                                                                                        beta,
#                                                                                        sensitivity_int,
#                                                                                        alpha,
#                                                                                        time_delay,
#                                                                                        2000)
# x_time = np.arange(len(s1_price))
# s1_log_return = np.divide(s1_log_return - np.mean(s1_log_return), np.std(s1_log_return))
# s2_log_return = np.divide(s2_log_return - np.mean(s2_log_return), np.std(s2_log_return))
#
# ax[0, 0].plot(np.arange(len(s1_price)), s1_price, label='S1', color='blue')
# ax[0, 0].plot(np.arange(len(s1_price)), s2_price, label='S2', color='red')
# ax[0, 0].set_title('Price Index')
# ax[0, 0].grid()
# ax[0, 0].set_xlabel('Time')
# ax[0, 0].set_ylabel('Price')
# ax[0, 0].legend()
#
# ax[0, 1].plot(np.arange(len(s1_log_return)), s1_log_return, label='S1', color='blue')
# ax[0, 1].plot(np.arange(len(s1_log_return)), s2_log_return, label='S2', color='red')
# ax[0, 1].set_title('Log Return')
# ax[0, 1].grid()
# ax[0, 1].set_xlabel('Time')
# ax[0, 1].set_ylabel('Log Return')
# ax[0, 1].legend()
#
# ax[1, 0].plot(np.arange(len(coupling_strat_tracker)), coupling_strat_tracker)
# ax[1, 0].set_title('Coupling Strat Usage Proportion')
# ax[1, 0].grid()
# ax[1, 0].set_xlabel('Time')
# ax[1, 0].set_ylabel('Proportion')
#
# plt.show()
#
# log_return_correlation = np.corrcoef(s1_log_return, s2_log_return)
# print(log_return_correlation)
#



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



###################################################
#
# # Settings
#
# shape = (512, 128)
# active_ratio = 17000 / (512 * 128)
# p_e = 0.0001
# p_d = 0.05
# p_h = 0.0493
# A = 1.8
# a = 2 * A
# h = 100
# beta = shape[0] ** 2 * shape[1] ** 2
#
# ###################################################
#
# # Plotting
#
# grid_hist, price, log_return, signal_hist, phase_hist = run_stock_price_simulation(
#     shape,
#     active_ratio,
#     p_e,
#     p_d,
#     p_h,
#     A,
#     a,
#     h,
#     beta,
#     1000)
#
# fig, ax = plt.subplots(2, 2, figsize=(12, 12))
#
# ax[0, 0].plot(price)
# ax[0, 0].set_title('Price Index')
# ax[0, 0].set_ylabel('Price')
# ax[0, 0].set_xlabel('Time')
# ax[0, 0].grid()
#
# ax[0, 1].plot(log_return)
# ax[0, 1].set_title('Log Return')
# ax[0, 1].set_ylabel('Log Return')
# ax[0, 1].set_xlabel('Time')
# ax[0, 1].grid()
#
# ax[1, 0].plot(phase_hist)
# ax[1, 0].set_title('Phase Values')
# ax[1, 0].set_ylabel('Phase Index')
# ax[1, 0].set_xlabel('Time')
#
# ax[1, 1].plot(signal_hist)
# ax[1, 1].set_title('Signal Function')
# ax[1, 1].set_xlabel('Time')
# ax[1, 1].grid()
#
# plt.show()
