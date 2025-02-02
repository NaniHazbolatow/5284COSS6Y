import numpy as np

from CAStockModel.model.utils.constants import DECREASING_SCALER_RISK_THD, INCREASING_SCALER_RISK_THD

def scale_risk_thd_by_previous_choice(
        previous_thd: np.ndarray,
        grid: np.ndarray,
        traders_following_coupling_strat: np.ndarray
) -> np.ndarray:
    """
    Updates the risk appetite of every previously active trader on the market, based on their previous decision. If their
    decision was to follow base strategy the risk appetite is going to increase by decreasing the risk appetite index,
    therefore making them more prone to consider S1 market price movement significant. If their decision was to follow
    coupling strategy, their risk appetite is decreased by increasing the risk appetite index, making them less prone to
    accept S1 market price movement significant.

    Args:
        previous_thd: Risk appetite index of every trader (2D numpy array same size as market grid)
        grid: 2D numpy array filled with 1, 0, -1 values representing traders on the market and their decisions.
        traders_following_coupling_strat: Filter (2D numpy array consisting of boolean values) for traders who chosen to follow coupling strategy

    Returns:
        Updated risk appetite index of every trader (2D numpy array same size as market grid)
    """
    new_thd = previous_thd.copy()

    # Increasing threshold for traders who followed coupling strategy
    new_thd[traders_following_coupling_strat] = INCREASING_SCALER_RISK_THD * new_thd[traders_following_coupling_strat]

    # Decreasing threshold for traders who did NOT follow coupling strategy
    traders_followed_base_strat = np.all([np.isin(grid, [1, -1]), ~traders_following_coupling_strat], axis=0)
    new_thd[traders_followed_base_strat] = DECREASING_SCALER_RISK_THD * new_thd[traders_followed_base_strat]

    return new_thd
