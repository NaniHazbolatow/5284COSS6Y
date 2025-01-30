import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from CAStockModel.main import run_coupled_stock_price_simulation
from CAStockModel.model.utils.utility_elements import convert_time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

################################################

# Settings

shape = (256, 256)
active_ratio = 0.25
p_e = 0.0001
p_d = 0.05
p_h = 0.0493
A = 1.8
a = 2 * A
h = 0
beta = shape[0] ** 2 * shape[1] ** 2
alpha = 100
time_delay = 25
self_organisation = False

################################################

phase_avg_cont = []
# sens_ub_values = [0.6, 0.8, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6]
sens_ub_values = [0.5, 0.75, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 6]

start_time = time.time()

logging.info(f"Phase Transition Plot Started    {convert_time(int(time.time() - start_time))}")

for ub in sens_ub_values:
    phase_cont = []
    for i in range(2):
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
            50
        )
        phase_cont.append(np.mean(np.abs(np.array(s2_phase))))
        logging.info(f"For {ub} upper bound simulation {i} done   {convert_time(int(time.time() - start_time))}")

    logging.info(f"UB {ub} is Finished with value {np.mean(np.array(phase_cont))}   {convert_time(int(time.time() - start_time))}")
    logging.info("_______________________________________")
    phase_avg_cont.append(np.mean(np.array(phase_cont)))

plt.figure(figsize=(8, 6))

plt.plot(sens_ub_values, phase_avg_cont, linewidth=2)

plt.grid()
plt.title("Phase Transition As a Function of Trader Decision Sensitivity Upper Bound")
plt.xlabel("Sensitivity Upper Bound")
plt.ylabel("Phase Transition Value")

plt.show()
