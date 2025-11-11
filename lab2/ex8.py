from ..common import Signal
from ..common import SignalType
import matplotlib.pyplot as plt
import numpy as np

value_range = np.arange(-np.pi / 2, np.pi / 2, 0.001)
sin_values = np.sin(value_range)
pade_sin_values = (value_range - 7 * value_range ** 3 / 60) / (1 + value_range ** 2 / 20)

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].plot(value_range, value_range)
axes[0].plot(value_range, sin_values)
axes[0].set_ylabel("value")
axes[0].grid(True)

axes[1].plot(value_range, sin_values - value_range)
axes[1].plot(value_range, pade_sin_values - sin_values)
axes[1].set_ylabel("error")
axes[1].grid(True)

abs_error_linear = np.abs(sin_values - value_range)
abs_error_pade = np.abs(pade_sin_values - sin_values)

tiny_val = 1e-18 
abs_error_linear[abs_error_linear < tiny_val] = np.nan
abs_error_pade[abs_error_pade < tiny_val] = np.nan

axes[2].plot(value_range, abs_error_linear)
axes[2].plot(value_range, abs_error_pade)
axes[2].set_yscale('log')
axes[2].set_xlabel("alpha value")
axes[2].set_ylabel("error")
axes[2].grid(True)

plt.show()
