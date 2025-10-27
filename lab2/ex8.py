from common import Signal
from common import SignalType
import matplotlib.pyplot as plt
import numpy as np

value_range = np.arange(-np.pi / 2, np.pi / 2, 0.001)
sin_values = np.sin(value_range)
pade_sin_values = (value_range - 7 * value_range ** 3 / 60) / (1 + value_range ** 2 / 20)

fig, axes = plt.subplots(2, 1, figsize=(10, 12))

axes[0].plot(value_range, value_range)
axes[0].plot(value_range, sin_values)
axes[1].plot(value_range, sin_values - value_range)
axes[1].set_xlabel("alpha value")
axes[1].set_ylabel("error")

axes[0].grid(True)
axes[1].grid(True)

axes[0].plot(value_range, pade_sin_values)
axes[1].plot(value_range, pade_sin_values - sin_values)

plt.show()
