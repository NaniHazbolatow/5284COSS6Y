import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from CAStockModel.main import run_coupled_stock_price_simulation
from CAStockModel.model.utils.utility_elements import convert_time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

################################################

# Model Settings

shape = (256, 256)
active_ratio = 0.25
p_e = 0.0001
p_d = 0.05
p_h = 0.0493
A = 1.8
a = 2 * A
h = 0
beta = shape[0] ** 2 * shape[1] ** 2
alpha = 10
self_organisation = True

# Statistics Settings

n_rerun = 100
simulation_len = 100

################################################

phase_avg_cont = []
sens_ub_values = [0.5, 0.75, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 6]
time_delays = [5, 10, 15, 20, 25, 30, 35]

start_time = time.time()

logging.info(f"Phase Transition Plot Started    {convert_time(int(time.time() - start_time))}")

# Containers for the data
simulation_results = []

# Simulating the data
for time_delay in time_delays:
    temp_results = []
    for ub in sens_ub_values:
        phase_cont = []
        for i in range(n_rerun):
            # Simulation
            _, _, _, _, _, _, _, _, _, s2_phase, _ = run_coupled_stock_price_simulation(
                shape,
                active_ratio,
                ub,
                p_e,
                p_d,
                p_h,
                A,
                a,
                h,
                beta,
                alpha,
                time_delay,
                self_organisation,
                simulation_len
            )

            # Appending the average absolute phase index value
            phase_cont.append(np.mean(np.abs(s2_phase)))
            logging.info(f"For {ub} upper bound simulation {i} done   {convert_time(int(time.time() - start_time))}")

        # Appending the results to the temporary container
        temp_results.append(phase_cont)

        logging.info(f"UB {ub} is Finished with value {np.mean(np.array(phase_cont))}   {convert_time(int(time.time() - start_time))}")
        logging.info("_______________________________________")

    # Appending the results to the temporary container
    simulation_results.append(np.array(temp_results))
    logging.info(f"Time Delay {time_delay} is Finished   {convert_time(int(time.time() - start_time))}")
    logging.info("______________________________________________________________________________")

# Creating the plot
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan']

plt.figure(figsize=(10, 6))

for i, current_data in enumerate(simulation_results):
    means = np.mean(current_data, axis=1)
    std_devs = np.std(current_data, axis=1, ddof=1)
    confidence_interval = 1.96 * std_devs / np.sqrt(n_rerun)

    plt.plot(sens_ub_values, means, label=time_delays[i], color=colors[i], linewidth=2)
    plt.fill_between(sens_ub_values, means - confidence_interval, means + confidence_interval,
                     color=colors[i], alpha=0.3)

plt.xlabel('Sensitivity Upper Bound')
plt.ylabel('Phase Index')
plt.title('Phase Transition Plot for Different Time Delays (alpha = 5%)')
plt.legend(title='Time Delay', facecolor='white', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
