from common import Signal
from common import SignalType

import numpy as np
import matplotlib.pyplot as plt
import math

fig, axes = plt.subplots(2, 1, figsize=(10, 12))
precision = 0.00001
duration = 0.1
sine_signal = Signal("Example Signal", 1, 80, 0, SignalType.SINE, duration, precision).with_sampling(1000)

print(sine_signal.samples)

length = len(sine_signal.get_function())

y = sine_signal.get_function() * math.e ** (-2 * np.pi * 1j * np.arange(0, 1, 1 / length))
print(y)
sine_signal.plot_signal(axes[0])
axes[1].plot(y.real, y.imag)
axes[1].set_xlim(-1, 1)
axes[1].set_ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()

