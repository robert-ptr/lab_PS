from ..common import Signal
from ..common import SignalType
import matplotlib.pyplot as plt 
import numpy as np

signals = [Signal("Signal 0", 1, 400, 0, SignalType.SINE, 0.32).with_sampling(5000),
           Signal("Signal 1", 1, 800, 0, SignalType.SINE, 0.03),
           Signal("Signal 2", 1, 240, 0, SignalType.SAWTOOTH, 0.03),
           Signal("Signal 3", 1, 300, 0, SignalType.SQUARE, 0.3)]

combined_signal = signals[1].get_function() + signals[2].get_function()
   
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

time_series = np.arange(0, 0.03, 0.00001)

axes[0].plot(time_series, signals[1].get_function())
axes[0].set_title(signals[1].name) 
axes[0].set_ylabel("Values")
axes[0].grid(True)

axes[1].plot(time_series, signals[2].get_function())
axes[1].set_title(signals[2].name)
axes[1].set_ylabel("Values")
axes[1].grid(True)

axes[2].plot(time_series, combined_signal)
axes[2].set_title("Combined Signal (Signal 1 + Signal 2)")
axes[2].set_ylabel("Values")
axes[2].set_xlabel("Time [s]")
axes[2].grid(True)

plt.tight_layout()
plt.show()
