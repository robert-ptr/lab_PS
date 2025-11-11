from ..common import Signal
from ..common import SignalType

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation, PillowWriter

fig, axes = plt.subplots(2, 3, figsize=(10, 12))
axes = axes.flatten()
precision = 0.00001
duration = 0.1
sine_signal = Signal("Example Signal", 1, 80, 0, SignalType.SINE, duration, precision).with_sampling(1000)

print(sine_signal.samples)

length = len(sine_signal.get_function())

omega_values = [1, 2, 5, 7, 80]
y_values = [None] * len(omega_values)
sine_signal.plot_signal(axes[0])

for i in range(len(omega_values)):
    axes[1 + i].set_aspect('equal', adjustable='box')
    y_values[i] = sine_signal.get_function() * math.e ** (-2 * np.pi * 1j * omega_values[i] * np.arange(0, 1, 1 / length))
    axes[1 + i].plot(y_values[i].real, y_values[i].imag)
    axes[1 + i].set_xlim(-1, 1)
    axes[1 + i].set_ylim(-1, 1)


plt.grid()
plt.show()

x = y_values[i].real
y = y_values[i].imag

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

line, = ax.plot([], [], linewidth=2)
point, = ax.plot([], [], 'o')

def update(i):
    line.set_data(x[:i], y[:i])
    point.set_data([x[i]], [y[i]])
    return line, point

frames = len(x)
anim = FuncAnimation(fig, update, frames=frames, interval=1, blit=False)

writer = PillowWriter(fps=60)
anim.save("signal.gif", writer=writer)

plt.show()
