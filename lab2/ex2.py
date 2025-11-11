from ..common import Signal
from ..common import SignalType
import numpy as np
import matplotlib.pyplot as plt

phase_list = [0, 3, 7, np.pi * 3 / 2]
#fig, ax = plt.subplots()

for phase in phase_list:
    sine_signal = Signal("Sine", 1, 170, phase, SignalType.SINE, 0.03).with_sampling(1000)
#    sine_signal.plot_signal(ax)

x = Signal("Sine", 1, 170, phase_list[3], SignalType.SINE, 0.03).get_function()

np.random.seed(42)
z = np.random.normal(0, 1, len(x))

norm_x = np.linalg.norm(x)
norm_z = np.linalg.norm(z)

snr_list = [0.1, 1, 10, 100]

fig, axes = plt.subplots(len(snr_list), 1, figsize=(10, 12))

for snr, ax in zip(snr_list, axes):
    gamma = norm_x / (norm_z * np.sqrt(snr))
    noise = gamma * z
    y = x + noise
    
    time_series = np.arange(0, 0.03, 0.00001)
    
    ax.plot(time_series, y)
    ax.plot(time_series, x, 'r--', alpha=0.7)
    ax.set_title("Signal with noise")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Values")
    ax.grid(True)

plt.show()
