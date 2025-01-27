import numpy as np
from scipy.stats import norm

def generate_correlated_values(
    u_range: tuple[float, float],
    r_range: tuple[int, int],
    rho: float,
    size: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two arrays: U as uniformly distributed floats and R as integers, correlated with correlation rho in the underlying normal space.
    Each dimension has a custom range.

    Parameters
    ----------
    u_range : (float, float)
        (lower_limit, upper_limit) for U.
    r_range : (int, int)
        (lower_limit, upper_limit) for R.
    rho : float
        Desired correlation between U and R in the underlying normal space.
        Must satisfy -1 <= rho <= 1.
    size : tuple
        Shape of the output arrays (e.g., (n, m)).

    Returns
    -------
    U, R : np.ndarray, np.ndarray
        Arrays of shape `size` containing the samples.
    """

    # Unpack the ranges
    low_u, high_u = u_range
    low_r, high_r = r_range
    
    # Safety checks
    if not (-1 <= rho <= 1):
        raise ValueError("Correlation rho must be in [-1, 1].")
    if low_u >= high_u:
        raise ValueError("U-range lower bound must be less than upper bound.")
    if low_r >= high_r:
        raise ValueError("R-range lower bound must be less than upper bound.")

    # 1. Construct the Cholesky factor for the 2x2 correlation matrix
    #    The correlation matrix is [[1, rho], [rho, 1]]
    L = np.array([
        [1.0,             0.0],
        [rho, np.sqrt(1 - rho**2)]
    ])

    # 2. Generate standard normal samples of shape (*size, 2)
    X = np.random.randn(*size, 2)

    # 3. Multiply by the transpose of L to get correlated normals.
    #    z will also have shape (*size, 2)
    z = X @ L.T  # shape: (*size, 2)

    # 4. Convert correlated normals to correlated uniforms in [0, 1]
    uv = norm.cdf(z)  # shape: (*size, 2)

    # 5. Scale to the desired ranges.
    #    For U, map u in [0,1] to [low_u, high_u] using:
    #       U = low_u + u * (high_u - low_u)
    #    For R, map u in [0,1] to [low_r, high_r] using:
    #       R = floor(low_r + u * (high_r - low_r + 1))
    scaled_u = low_u + uv[..., 0] * (high_u - low_u)
    scaled_r = low_r + uv[..., 1] * (high_r - low_r + 1)

    # 6. Floor and cast R to integers
    R = np.floor(scaled_r).astype(int)

    # 7. Clip final values to ensure we remain in [low_r, high_r].
    R_clipped = np.clip(R, low_r, high_r)

    return scaled_u, R_clipped


import numpy as np

def compute_moving_average_and_volatility(price_index: np.ndarray, look_back: np.ndarray):
    # Ensure look_back is an array for element-wise operations
    look_back = np.asarray(look_back)
    
    # Length of the price index
    N = len(price_index)
    
    # Create a mask for valid look_back values
    valid_mask = look_back <= N
    
    # Initialize results with NaNs to handle invalid look_back values
    moving_average = np.full_like(look_back, np.nan, dtype=float)
    volatility = np.full_like(look_back, np.nan, dtype=float)
    
    # Only compute for valid look_back values
    if np.any(valid_mask):
        valid_look_back = look_back[valid_mask]
        
        # Pre-compute cumulative sums
        price_sum = np.concatenate(([0], np.cumsum(price_index)))
        price_squared_sum = np.concatenate(([0], np.cumsum(price_index * price_index)))
        
        # Compute look-back values for sum of p and sum of p^2
        sum_d = price_sum[N] - price_sum[N - valid_look_back]
        sum_d_sq = price_squared_sum[N] - price_squared_sum[N - valid_look_back]
        
        # Compute moving average for valid look_back
        look_back_div = valid_look_back.astype(float)  # Ensure float division
        moving_average[valid_mask] = sum_d / look_back_div
        
        # Compute volatility for valid look_back
        volatility[valid_mask] = np.sqrt((sum_d_sq / look_back_div) - (moving_average[valid_mask] ** 2))
    
    return moving_average, volatility


def construct_signal(prices, moving_average, volatility, risk_appetite):
    # Signal construction based on bollinger band
    if np.any(np.isnan(moving_average)) or np.any(np.isnan(volatility)):
        return np.zeros_like(moving_average)

    # Calculate the Bollinger Band thresholds
    upper_band = moving_average + (risk_appetite * volatility)
    lower_band = moving_average - (risk_appetite * volatility)

    # Construct the signal matrix
    signal = np.zeros_like(moving_average)
    signal[prices[-1] > upper_band] = -1  # Short signal
    signal[prices[-1] < lower_band] = 1   # Long signal

    return signal

