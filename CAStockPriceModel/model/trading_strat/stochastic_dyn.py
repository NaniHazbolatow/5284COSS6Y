from typing import Tuple

import numpy as np
from numba import njit

from CAStockModel.model.base_model.stochastic_dyn import apply_stochastic_dynamics
from CAStockModel.model.utils.utility_elements import order_clusters


def stochastic_dynamics_coupling_strategy(
        grid: np.ndarray,
        trader_decision_sensitivity: np.ndarray,
        clusters: np.ndarray,
        A: float,
        a: float,
        h: float,
        price_index: np.ndarray,
        time_delay: float,
        alpha: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the new grid by simulating decisions of the traders whether to sell or buy, who choose to be active in
    the next time step. Every trader decides whether they want to follow the coupling strategy or the more conservative
    base strategy, which is implemented in the stochastic_dyn module under the base_model. The decision is going to be
    based on the analysis of the S1 stock price and making assumption whether the normalised deviation of the most recent
    value high enough (in absolute value) to imply a significant market movement or just stochastic noise. If the coupling
    strategy is chosen, the direction of the deviation deterministically determines whether to buy or sell. Traders are
    going to work with different window of information of the S1 price index. The time delay of the real time data is
    determined by the cluster size the trader belongs to. The bigger the cluster size the more recent the time its time
    series is.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        trader_decision_sensitivity: 2D numpy array (same size as grid) where every value represents the risk threshold for the trader at the corresponding index in the market grid.
        clusters: 3D numpy array where every slice along the 0 axis consists of filter of traders belonging to the same clusters.
        A: General cluster level trust scaler (fixed model parameter)
        a: One to one trust scaler (fixed model parameter)
        h: External decision altering force scaler (fixed model parameter)
        price_index: S1 price index (1D numpy array)
        time_delay: The time lengths with the S1 price index is delayed with for traders in smaller clusters. (The delay can be 0, 1, 2, 3 times the time delay value.)
        alpha: Length of the time window of S1 price index (same for every trader)

    Returns:
        The grid updated with the decisions and filter (2D numpy array with boolean values) for traders who choose to follow the coupling strategy.
    """
    new_grid = grid.copy()

    # Calculating strategy decision related data
    delayed_price_info = _calculate_price_info(price_index, time_delay, alpha)
    average_price, std_price = _calculate_price_reference_moments(delayed_price_info)

    # Setting strategy decisions
    trader_info_delay_categories = _categorise_traders_by_cluster_size(grid, clusters)
    traders_following_coupling_strat, couple_strat_decisions = _decide_trader_strategy(grid,
                                                                                       delayed_price_info,
                                                                                       trader_info_delay_categories,
                                                                                       trader_decision_sensitivity,
                                                                                       average_price,
                                                                                       std_price)

    # Creating the t + 1 grid
    _update_traders(new_grid, clusters, traders_following_coupling_strat, couple_strat_decisions, A, a, h)

    return new_grid, traders_following_coupling_strat


@njit
def _calculate_price_info(price_index: np.ndarray, time_delay: float, alpha: int) -> np.ndarray:
    """
    Creates 2D numpy array in which every row corresponds to a time series of the price index which is available
    to the different groups of traders, based on their info delay category.

    Args:
        price_index: S1 price index (1D numpy array)
        time_delay: The time lengths with the S1 price index is delayed with for traders in smaller clusters. (The delay can be 0, 1, 2, 3 times the time delay value.)
        alpha: Length of the time window of S1 price index (same for every trader)

    Returns:
        Potential four S1 price time series (2D numpy array)
    """

    if len(price_index) < alpha + 3 * time_delay:
        raise ValueError(f"Price array is not long enough. It is {len(price_index)} long, "
                          f"while it should be at least {alpha + 3 * time_delay}.")

    # Snipping the price index that can have effect on the strategy decisions
    relevant_price_index = price_index[::-1][:alpha + time_delay * 3]

    return np.lib.stride_tricks.sliding_window_view(relevant_price_index, alpha)[::time_delay][:4, ::-1]


def _calculate_price_reference_moments(price_info: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the mean and the standard deviation of all the price info's of the 4 delay category.

    Args:
        price_info: Potential four S1 price time series (2D numpy array)

    Returns:
        The mean and standard deviation for all the price info's of the 4 delay category
    """

    if price_info.shape[0] != 4:
        raise ValueError(f"Price info has {price_info.shape[0]} rows, while it should have 4 aligned with the four "
                          f"info delay categories of the traders.")

    return np.mean(price_info, axis=1), np.std(price_info, axis=1)


def _categorise_traders_by_cluster_size(grid: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Assigns a category index from the range [0, 3] to every trader, based on the clusters size they belong.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        clusters: 3D numpy array where every slice along the 0 axis consists of filter of traders belonging to the same clusters.

    Returns:
        The info (2D numpy array same size as market grid) of which trader gets delayed by what scaler of the time delay value
    """
    trader_categories = np.zeros_like(grid)

    # Sort 'clusters' input by the size of the clusters
    ordered_clusters = order_clusters(clusters)

    # Check whether there is enough cluster to assign all the 4 categories
    if len(ordered_clusters) > 4:
        # Calculate which cluster gets delayed by 0, t, 2*t, 3*t
        n = len(ordered_clusters)
        q1, q2, q3 = n // 4 + 1, n // 2 + 1, n // 4 * 3 + 1

        # Filters for traders in different delay categories
        delay_3_traders = np.any(ordered_clusters[:q1], axis=0)
        delay_2_traders = np.any(ordered_clusters[q1:q2], axis=0)
        delay_1_traders = np.any(ordered_clusters[q2:q3], axis=0)

        trader_categories[delay_3_traders] = 3
        trader_categories[delay_2_traders] = 2
        trader_categories[delay_1_traders] = 1

        return trader_categories

    else:
        # If all the active traders are grouped in 4 or fewer clusters, we consider those clusters big enough to have instant information from S1 price index
        return trader_categories


def _decide_trader_strategy(
        grid: np.ndarray,
        price_info: np.ndarray,
        trader_delay_cat: np.ndarray,
        trader_decision_sensitivity: np.ndarray,
        avg_prices: np.ndarray,
        std_prices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decides whether traders would follow base or coupling strategy and whether to buy or sell if the latter one is applicable.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        price_info: Potential four S1 price time series (2D numpy array)
        trader_delay_cat: The info (2D numpy array same size as market grid) of which trader gets delayed by what scaler of the time delay value
        trader_decision_sensitivity: 2D numpy array (same size as grid) where every value represents the risk threshold for the trader at the corresponding index in the market grid.
        avg_prices: Average of the four possible time series of the S1 price index
        std_prices: Standard Deviation of the four possible time series of the S1 price index

    Returns:
        Filter (2D numpy array containing boolean values) for traders following coupling strategy and their final decision (2D numpy array) to buy or sell.
    """

    # Calculate the individual reference moments of the S1 Price index
    ref_mean = avg_prices[trader_delay_cat]
    ref_std = std_prices[trader_delay_cat]

    # Calculating the relative S1 Price index deviation from the reference point (ts average)
    closing_prices = price_info[:, -1]
    ref_price_point = closing_prices[trader_delay_cat]
    price_rel_deviation = np.divide(ref_price_point - ref_mean, ref_std)

    # Decide whether the trader follows coupling or base strategy
    traders_following_coupling_strat = np.all(np.stack((np.abs(price_rel_deviation) > trader_decision_sensitivity, np.isin(grid, [1, -1])), axis=0), axis=0)

    # Decide whether to buy or sell if coupling strategy is followed
    couple_strategy_decisions = np.where(price_rel_deviation > trader_decision_sensitivity, 1, -1)

    return traders_following_coupling_strat, couple_strategy_decisions


def _update_traders(
        grid: np.ndarray,
        clusters: np.ndarray,
        traders_following_coupling_strat: np.ndarray,
        coupling_start_decision: np.ndarray,
        A: float,
        a: float,
        h: float
) -> None:
    """
    Finalising the trading direction (buy or sell) for all the traders for t + 1, by updating the existing market grid.

    Args:
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        clusters: 3D numpy array where every slice along the 0 axis consists of filter of traders belonging to the same clusters.
        traders_following_coupling_strat: Filter (2D numpy array containing boolean values) for traders following coupling strategy.
        coupling_start_decision: Decisions made by the traders chosen to follow coupling strategy. (2D numpy array)
        A: General cluster level trust scaler (fixed model parameter)
        a: One to one trust scaler (fixed model parameter)
        h: External decision altering force scaler (fixed model parameter)

    Returns:
        None
    """

    # Calculating the decision for traders who chose to follow base strategy
    base_strat_decisions = apply_stochastic_dynamics(grid, clusters, A, a, h)
    traders_following_base_strategy = ~traders_following_coupling_strat
    grid[traders_following_base_strategy] = base_strat_decisions[traders_following_base_strategy]

    # Assigning the decision where coupling strategy was chosen
    grid[traders_following_coupling_strat] = coupling_start_decision[traders_following_coupling_strat]
