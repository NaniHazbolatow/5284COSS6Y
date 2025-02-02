import os

import matplotlib.pyplot as plt

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

(_, s1_price, s1_log_ret, s1_signal, s1_phase,
 _, s2_price, s2_log_ret, s2_signal, s2_phase, _) = run_coupled_stock_price_simulation(
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
   10
)

################################################

# Plotting

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
plt.savefig(os.path.join(figure_dir, 'model_result_plot.png'), dpi=500, bbox_inches='tight', format='png')
plt.show()
