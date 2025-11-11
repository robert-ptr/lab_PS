from ..common import Signal
from ..common import SignalType
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

sine_signal = Signal("Sine Signal 0", 2, 300, 3, SignalType.SINE, 0.01)
cos_signal = Signal("Cosine Signal 0", 2, 300, 3 - np.pi / 2, SignalType.COSINE, 0.01)
sine_signal.plot_signal(ax1)
cos_signal.plot_signal(ax2)

plt.show()
