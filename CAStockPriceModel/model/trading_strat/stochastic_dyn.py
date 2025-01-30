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
    """Creates 2D numpy array in which every row corresponds to a time series of the price index which is available
    to the different groups of traders, based on their info delay category"""

    if len(price_index) < alpha + 3 * time_delay:
        raise ValueError(f"Price array is not long enough. It is {len(price_index)} long, "
                          f"while it should be at least {alpha + 3 * time_delay}.")

    # Snipping the price index that can have effect on the strategy decisions
    relevant_price_index = price_index[::-1][:alpha + time_delay * 3]

    return np.lib.stride_tricks.sliding_window_view(relevant_price_index, alpha)[::time_delay][:4, ::-1]


def _calculate_price_reference_moments(price_info: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the mean and the standard deviation of all the price info's of the 4 delay category"""

    if price_info.shape[0] != 4:
        raise ValueError(f"Price info has {price_info.shape[0]} rows, while it should have 4 aligned with the four "
                          f"info delay categories of the traders.")

    return np.mean(price_info, axis=1), np.std(price_info, axis=1)


def _categorise_traders_by_cluster_size(grid: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Assigns a category index from the range [0, 3] to every trader, based on the clusters size they belong"""
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
    """Decides whether traders would follow base or coupling strategy and whether to buy or sell if the latter one is applicable"""

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
    """Finalising the trading direction (buy or sell) for all the traders for t + 1"""

    # Calculating the decision for traders who chose to follow base strategy
    base_strat_decisions = apply_stochastic_dynamics(grid, clusters, A, a, h)
    traders_following_base_strategy = ~traders_following_coupling_strat
    grid[traders_following_base_strategy] = base_strat_decisions[traders_following_base_strategy]

    # Assigning the decision where coupling strategy was chosen
    grid[traders_following_coupling_strat] = coupling_start_decision[traders_following_coupling_strat]
