from ..common import SignalType
from ..common import Signal
import numpy as np
import matplotlib.pyplot as plt

duration = 0.5 # seconds

signal0 = Signal("Signal 0", amplitude=1, freq=10, phase=0, signal_type=SignalType.SINE, duration=duration).with_sampling(12)
signal1 = Signal("Signal 1", amplitude=1, freq=2, phase=np.pi, signal_type=SignalType.SINE, duration=duration).with_sampling(12)
signal2 = Signal("Signal 2", amplitude=1, freq=26, phase=np.pi, signal_type=SignalType.SINE, duration=duration).with_sampling(12)

fig, axes = plt.subplots(3, figsize=(10, 6))

signal0.plot_signal(axes[0])
signal1.plot_signal(axes[1])
signal2.plot_signal(axes[2])

plt.show()