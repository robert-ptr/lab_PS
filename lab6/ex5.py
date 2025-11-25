import numpy as np
import matplotlib.pyplot as plt
from ..common import Signal
from ..common import SignalType

sine_signal = Signal("Sine Signal", 1, 100, 0, SignalType.SINE, 1)

def create_rectangular_window(N):
    return np.ones(N)

def create_hanning_window(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / N))

rect_window = create_rectangular_window(len(sine_signal.get_function()))
hanning_window = create_hanning_window(len(sine_signal.get_function()))

y1 = sine_signal.get_function() * rect_window
y2 = sine_signal.get_function() * hanning_window

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(np.arange(0, 1, 0.00001), y1)
axes[1].plot(np.arange(0, 1, 0.00001), y2)

plt.show()