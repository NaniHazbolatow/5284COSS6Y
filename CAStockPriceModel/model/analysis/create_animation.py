import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from CAStockModel.main import run_coupled_stock_price_simulation
from CAStockModel.model.utils.constants import figure_dir


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

# Run Simulation

s1_grid, _, _, _, _, s2_grid, _, _, _, _, _ = run_coupled_stock_price_simulation(
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
    10000
)

################################################

# Animation

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
ani.save(os.path.join(figure_dir, 'grid_animation.html'), writer='html', fps=5)

# Display the animation
plt.tight_layout()
plt.show()
