import time
import numpy as np
from scipy.stats import norm
from scipy.ndimage import label
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------------------------
# Helper function for time conversion
# -----------------------------------------------------------------------------
def convert_time(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"

# -----------------------------------------------------------------------------
# Market grid and cluster functions
# -----------------------------------------------------------------------------
def initialise_market_grid(shape: tuple, active_ratio: float) -> np.ndarray:
    total_cells = shape[0] * shape[1]
    active_indices = np.random.choice(total_cells, replace=False, size=int(active_ratio * total_cells))
    one_d_grid = np.zeros(total_cells, dtype=np.int8)
    one_d_grid[active_indices] = np.random.choice([1, -1], size=len(active_indices), replace=True)
    two_d_grid = one_d_grid.reshape(shape)
    return two_d_grid

def find_clusters(grid: np.ndarray) -> np.ndarray:
    active_cells = (grid == 1) | (grid == -1)
    # Define von Neumann neighborhood structure
    von_neumann_structure = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]])
    clusters, num_clusters = label(active_cells, structure=von_neumann_structure)
    # Create a boolean mask for each cluster (cluster ids start at 1)
    cluster_filters = np.array([(clusters == cluster_id) for cluster_id in range(1, num_clusters + 1)])
    return cluster_filters

def calculate_log_return(grid: np.ndarray, clusters: np.ndarray, beta: float) -> float:
    trader_weighted_sum = np.sum([np.sum(cluster) * np.sum(grid[cluster]) for cluster in clusters])
    log_return = trader_weighted_sum / beta
    return log_return

# -----------------------------------------------------------------------------
# Percolation dynamics functions (Homogenous & Heterogenous)
# -----------------------------------------------------------------------------
def apply_percolation_dyn(grid: np.ndarray, p_e: float, p_d: float, p_h: float) -> np.ndarray:
    neighbours = _create_neighbours(grid)
    mask_1, mask_3, mask_4 = _create_masks(grid, neighbours)
    n_inactive_neighbour, n_active_neighbour = _count_neighbours(neighbours)
    prob_1, prob_3, prob_4 = _calculate_probabilities(n_inactive_neighbour, n_active_neighbour, p_e, p_d, p_h)
    eff_filter_1, eff_filter_3, eff_filter_4 = _create_effective_filters(
        grid, prob_1, prob_3, prob_4, mask_1, mask_3, mask_4
    )
    new_grid = _create_new_grid(grid, eff_filter_1, eff_filter_3, eff_filter_4)
    return new_grid

def _create_neighbours(grid: np.ndarray) -> np.ndarray:
    padded_grid = np.pad(grid, pad_width=1, mode="constant", constant_values=0)
    top_neighbours = padded_grid[:-2, 1:-1]
    bottom_neighbours = padded_grid[2:, 1:-1]
    left_neighbours = padded_grid[1:-1, :-2]
    right_neighbours = padded_grid[1:-1, 2:]
    return np.stack([top_neighbours, bottom_neighbours, left_neighbours, right_neighbours], axis=0)

def _create_masks(grid: np.ndarray, neighbours: np.ndarray) -> tuple:
    # Type 1: Inactive cell with all neighbours inactive
    mask_1 = (grid == 0) & np.all(neighbours == 0, axis=0)
    # Type 3: Inactive cell with at least one active neighbour
    mask_3 = (grid == 0) & np.any(np.isin(neighbours, [1, -1]), axis=0)
    # Type 4: Active cell with at least one inactive neighbour
    mask_4 = (np.isin(grid, [1, -1])) & np.any(neighbours == 0, axis=0)
    return mask_1, mask_3, mask_4

def _count_neighbours(neighbours: np.ndarray) -> tuple:
    n_inactive_neighbour = np.sum(neighbours == 0, axis=0)
    n_active_neighbour = np.sum(np.isin(neighbours, [1, -1]), axis=0)
    return n_inactive_neighbour, n_active_neighbour

@njit
def _calculate_probabilities(
    inactive_neighbour: np.ndarray,
    active_neighbour: np.ndarray,
    p_e: float,
    p_d: float,
    p_h: float
) -> tuple:
    prob_1 = p_e
    prob_3 = 1 - ((1 - p_h) ** active_neighbour) * (1 - p_e)
    prob_4 = 1 - ((1 - p_d) ** inactive_neighbour)
    return prob_1, prob_3, prob_4

@njit
def _create_effective_filters(
    grid: np.ndarray,
    prob_1: float,
    prob_3: float,
    prob_4: float,
    mask_1: np.ndarray,
    mask_3: np.ndarray,
    mask_4: np.ndarray
) -> tuple:
    prob_grid = np.random.random(size=(grid.shape[0], grid.shape[1]))
    type_1_filter = (prob_grid < prob_1) & mask_1
    type_3_filter = (prob_grid < prob_3) & mask_3
    type_4_filter = (prob_grid < prob_4) & mask_4
    return type_1_filter, type_3_filter, type_4_filter

def _create_new_grid(
    grid: np.ndarray,
    eff_filter_1: np.ndarray,
    eff_filter_3: np.ndarray,
    eff_filter_4: np.ndarray
) -> np.ndarray:
    new_grid = grid.copy()
    new_grid[eff_filter_1] = np.random.choice([1, -1], size=np.sum(eff_filter_1), replace=True)
    new_grid[eff_filter_3] = np.random.choice([1, -1], size=np.sum(eff_filter_3), replace=True)
    new_grid[eff_filter_4] = 0
    return new_grid

def _create_new_grid_heterogenous(
    grid: np.ndarray,
    signal_matrix: np.ndarray,
    eff_filter_1: np.ndarray,
    eff_filter_3: np.ndarray,
    eff_filter_4: np.ndarray
) -> np.ndarray:
    new_grid = grid.copy()
    new_grid[eff_filter_1] = signal_matrix[eff_filter_1]
    new_grid[eff_filter_3] = np.random.choice([1, -1], size=np.sum(eff_filter_3), replace=True)
    new_grid[eff_filter_4] = 0
    return new_grid

def _calculate_probabilities_heterogenous(
    inactive_neighbour: np.ndarray,
    active_neighbour: np.ndarray,
    p_d: float,
    p_h: float,
) -> tuple:
    prob_3 = 1 - (1 - p_h) ** active_neighbour
    prob_4 = 1 - (1 - p_d) ** inactive_neighbour
    return prob_3, prob_4

def _create_effective_filters_heterogenous(
    grid: np.ndarray,
    prob_3: np.ndarray,
    prob_4: np.ndarray,
    mask_1: np.ndarray,
    mask_3: np.ndarray,
    mask_4: np.ndarray,
) -> tuple:
    prob_grid = np.random.random(size=grid.shape)
    # For Type 1, we use the signal matrix directly (mask_1 acts as the filter)
    type_1_filter = mask_1
    # Probabilistic filters for Types 3 and 4
    type_3_filter = (prob_grid < prob_3) & mask_3
    type_4_filter = (prob_grid < prob_4) & mask_4
    return type_1_filter, type_3_filter, type_4_filter

def apply_percolation_dyn_heterogenous(
    grid: np.ndarray, p_d: float, p_h: float, signal_matrix: np.ndarray
) -> np.ndarray:
    neighbours = _create_neighbours(grid)
    mask_1, mask_3, mask_4 = _create_masks(grid, neighbours)
    n_inactive_neighbour, n_active_neighbour = _count_neighbours(neighbours)
    prob_3, prob_4 = _calculate_probabilities_heterogenous(n_inactive_neighbour, n_active_neighbour, p_d, p_h)
    eff_filter_1, eff_filter_3, eff_filter_4 = _create_effective_filters_heterogenous(
        grid, prob_3, prob_4, mask_1, mask_3, mask_4
    )
    new_grid = _create_new_grid_heterogenous(
        grid, signal_matrix, eff_filter_1.astype(bool), eff_filter_3, eff_filter_4
    )
    return new_grid

# -----------------------------------------------------------------------------
# Stochastic dynamics functions
# -----------------------------------------------------------------------------
def apply_stochastic_dynamics(grid: np.ndarray, clusters: np.ndarray, A: float, a: float, h: float) -> np.ndarray:
    new_grid = grid.copy()
    for current_cluster in clusters:
        cluster_elements = grid[current_cluster]
        n = np.sum(current_cluster)
        A_ij, h_i = _simulate_connections(n, A, a, h)
        probabilities = _calculate_buying_probs(n, cluster_elements, A_ij, h_i)
        new_grid[current_cluster] = _cluster_value_upgrade(probabilities)
    return new_grid

@njit
def _simulate_connections(n: int, A: float, a: float, h: float) -> tuple:
    xi_k = np.random.uniform(-1, 1)
    nu_ij = np.random.uniform(-1, 1, size=(n, n))
    zeta_i = np.random.uniform(-1, 1, size=n)
    A_ij = A * xi_k + a * nu_ij
    h_i = h * zeta_i
    return A_ij, h_i

@njit
def _calculate_buying_probs(n: int, cluster_elements: np.ndarray, A_ij: np.ndarray, h_i: np.ndarray) -> np.ndarray:
    I_i = np.sum(A_ij * cluster_elements, axis=1) / n + h_i
    return 1 / (1 + np.exp(-2 * I_i))

@njit
def _cluster_value_upgrade(probabilities: np.ndarray) -> np.ndarray:
    return np.where(np.random.random(len(probabilities)) > probabilities, -1, 1)

# -----------------------------------------------------------------------------
# Functions for generating correlated values and signals
# -----------------------------------------------------------------------------
def generate_correlated_values(
    u_range: tuple[float, float],
    r_range: tuple[int, int],
    rho: float,
    size: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    low_u, high_u = u_range
    low_r, high_r = r_range
    if not (-1 <= rho <= 1):
        raise ValueError("Correlation rho must be in [-1, 1].")
    if low_u >= high_u:
        raise ValueError("U-range lower bound must be less than upper bound.")
    if low_r >= high_r:
        raise ValueError("R-range lower bound must be less than upper bound.")

    # Construct Cholesky factor for the 2x2 correlation matrix
    L = np.array([
        [1.0, 0.0],
        [rho, np.sqrt(1 - rho**2)]
    ])

    # Generate standard normal samples of shape (*size, 2)
    X = np.random.randn(*size, 2)
    # Obtain correlated normals
    z = X @ L.T
    # Convert to correlated uniforms in [0, 1]
    uv = norm.cdf(z)
    # Scale to desired ranges
    scaled_u = low_u + uv[..., 0] * (high_u - low_u)
    scaled_r = low_r + uv[..., 1] * (high_r - low_r + 1)
    R = np.floor(scaled_r).astype(int)
    R_clipped = np.clip(R, low_r, high_r)
    return scaled_u, R_clipped

def compute_moving_average_and_volatility(price_index: np.ndarray, look_back: np.ndarray):
    look_back = np.asarray(look_back)
    N = len(price_index)
    valid_mask = look_back <= N
    moving_average = np.full(look_back.shape, np.nan, dtype=float)
    volatility = np.full(look_back.shape, np.nan, dtype=float)
    if np.any(valid_mask):
        valid_look_back = look_back[valid_mask]
        price_sum = np.concatenate(([0], np.cumsum(price_index)))
        price_squared_sum = np.concatenate(([0], np.cumsum(price_index * price_index)))
        sum_d = price_sum[N] - price_sum[N - valid_look_back]
        sum_d_sq = price_squared_sum[N] - price_squared_sum[N - valid_look_back]
        look_back_div = valid_look_back.astype(float)
        moving_average[valid_mask] = sum_d / look_back_div
        volatility[valid_mask] = np.sqrt((sum_d_sq / look_back_div) - (moving_average[valid_mask] ** 2))
    return moving_average, volatility

def construct_signal(prices, moving_average, volatility, risk_appetite):
    """
    Constructs a signal matrix based on Bollinger Bands.
    
    For any cell where moving_average or volatility is NaN, the signal will be 0.
    For cells with valid data:
      - If the last price is above the upper threshold, the signal is -1 (short).
      - If the last price is below the lower threshold, the signal is 1 (long).
    """
    # Initialize the signal array to 0 with the same shape as moving_average.
    signal = np.zeros_like(moving_average)
    
    # Create a mask that is True for cells where both moving_average and volatility are valid.
    valid = ~(np.isnan(moving_average) | np.isnan(volatility))
    
    if np.any(valid):
        # Extract the valid entries (these become a 1D array).
        valid_ma   = moving_average[valid]
        valid_vol  = volatility[valid]
        valid_risk = risk_appetite[valid]
        
        # Compute the Bollinger band thresholds for the valid cells.
        upper_band = valid_ma + valid_risk * valid_vol
        lower_band = valid_ma - valid_risk * valid_vol
        
        # Use the last price from the prices array (a scalar).
        last_price = prices[-1]
        
        # Create a temporary 1D array to hold the signals for valid entries.
        valid_signal = np.zeros_like(valid_ma, dtype=signal.dtype)
        valid_signal[last_price > upper_band] = -1  # Short signal
        valid_signal[last_price < lower_band] = 1   # Long signal
        
        # Write the computed valid signals back into the full signal array.
        signal[valid] = valid_signal
        
    # Cells that are not valid remain 0.
    return signal


# -----------------------------------------------------------------------------
# Main simulation function
# -----------------------------------------------------------------------------
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
):
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
        # --- Simulation 1: Complete Homogenous ---
        grid_history_1.append(apply_percolation_dyn(grid_history_1[-1], p_e, p_d, p_h))
        clusters = find_clusters(grid_history_1[-1])
        grid_history_1[-1] = apply_stochastic_dynamics(grid_history_1[-1], clusters, A, a, h)
        log_return_1 = calculate_log_return(grid_history_1[-1], clusters, beta)
        log_returns_1.append(log_return_1)
        price_index_1.append(price_index_1[-1] * np.exp(log_return_1))

        # --- Simulation 2: Complete Heterogenous ---
        ma, vol = compute_moving_average_and_volatility(np.array(price_index_1), look_back)
        # (Optional) Debug: print moving averages and volatilities
        signal = construct_signal(np.array(price_index_1), ma, vol, risk_apt)
        grid_history_2.append(apply_percolation_dyn_heterogenous(grid_history_2[-1], p_d, p_h, signal))
        clusters = find_clusters(grid_history_2[-1])
        grid_history_2[-1] = apply_stochastic_dynamics(grid_history_2[-1], clusters, A, a, h)
        log_return_2 = calculate_log_return(grid_history_2[-1], clusters, beta)
        log_returns_2.append(log_return_2)
        price_index_2.append(price_index_2[-1] * np.exp(log_return_2))

        if i % 100 == 0:
            print(f"{i}th simulation DONE   {convert_time(int(time.time() - start_time))}")

    print(f"Simulation time is   {convert_time(int(time.time() - start_time))}")
    return (np.array(grid_history_1), np.array(price_index_1), np.array(log_returns_1),
            np.array(grid_history_2), np.array(price_index_2), np.array(log_returns_2))

# -----------------------------------------------------------------------------
# Settings and simulation run
# -----------------------------------------------------------------------------
shape = (50, 50)
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

# -----------------------------------------------------------------------------
# Plotting and animation
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
vmin, vmax = -1, 1  # Fixed value range for both grids

# Initialize grid plots
image1 = axs[0, 0].imshow(grid_history_1[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
axs[0, 0].set_title("Simulation 1: Grid")
image2 = axs[0, 1].imshow(grid_history_2[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
axs[0, 1].set_title("Simulation 2: Grid")

# Initialize price index plots
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

def update(frame):
    image1.set_array(grid_history_1[frame])
    image2.set_array(grid_history_2[frame])
    line1.set_data(range(frame + 1), price_index_1[:frame + 1])
    line2.set_data(range(frame + 1), price_index_2[:frame + 1])
    return image1, image2, line1, line2

ani = FuncAnimation(fig, update, frames=sim_number, interval=100, blit=True)
plt.tight_layout()
plt.show()
