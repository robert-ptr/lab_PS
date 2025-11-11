from ..common import Signal
from ..common import SignalType
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

signal0 = Signal("Signal 0", 1, 110, 0, SignalType.SINE, 0.1).with_sampling(1000)
signal1 = Signal("Signal 1", 1, 110, 0, SignalType.SINE, 0.1).with_sampling(250)
signal2 = Signal("Signal 2", 1, 110, 0, SignalType.SINE, 0.1).with_sampling(62.5)

signal0.plot_signal(axes[0])
signal1.plot_signal(axes[1])
signal2.plot_signal(axes[2])

plt.tight_layout()
plt.show()

