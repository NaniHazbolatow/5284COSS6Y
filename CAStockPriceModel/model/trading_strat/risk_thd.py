from typing import Tuple

import numpy as np

from CAStockModel.model.utils.constants import (DECREASING_SCALER_RISK_THD, INCREASING_SCALER_RISK_THD,
                                                SENSITIVITY_LOWER_BOUND)

def scale_risk_thd_by_previous_choice(
        previous_thd: np.ndarray,
        grid: np.ndarray,
        traders_following_coupling_strat: np.ndarray
) -> np.ndarray:
    new_thd = previous_thd.copy()

    # Increasing threshold for traders who followed coupling strategy
    new_thd[traders_following_coupling_strat] = INCREASING_SCALER_RISK_THD * new_thd[traders_following_coupling_strat]

    # Decreasing threshold for traders who did NOT follow coupling strategy
    traders_followed_base_strat = np.all([np.isin(grid, [1, -1]), ~traders_following_coupling_strat], axis=0)
    new_thd[traders_followed_base_strat] = DECREASING_SCALER_RISK_THD * new_thd[traders_followed_base_strat]

    return new_thd


def sample_risk_thd_by_previous_choice(
        previous_sensitivity_ub: np.ndarray,
        previous_thd: np.ndarray,
        grid: np.ndarray,
        traders_following_coupling_strat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    new_thd = previous_thd.copy()
    new_sensitivity_ub = previous_sensitivity_ub.copy()

    # Increasing sensitivity upper bound for traders who followed coupling strategy
    new_sensitivity_ub[traders_following_coupling_strat] = (INCREASING_SCALER_RISK_THD *
                                                            new_sensitivity_ub[traders_following_coupling_strat])

    # Resampling the sensitivities for traders who followed coupling strategy
    new_thd[traders_following_coupling_strat] = np.random.uniform(SENSITIVITY_LOWER_BOUND,
                                                                  new_sensitivity_ub[traders_following_coupling_strat])

    # Decreasing sensitivity upper bound for traders who did NOT follow coupling strategy
    traders_followed_base_strat = np.all(np.isin(grid, [1, -1]), ~traders_following_coupling_strat)
    new_sensitivity_ub[traders_followed_base_strat] = (DECREASING_SCALER_RISK_THD *
                                                       new_sensitivity_ub[traders_followed_base_strat])

    # Resampling the sensitivities for traders who did NOT follow coupling strategy
    new_thd[traders_followed_base_strat] = np.random.uniform(SENSITIVITY_LOWER_BOUND,
                                                             new_sensitivity_ub[traders_followed_base_strat])

    return new_thd, new_sensitivity_ub
