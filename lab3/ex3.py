from common import Signal
from common import SignalType

import matplotlib.pyplot as plt
import numpy as np
import math

duration = 0.3
precision = 0.001

signal0 = Signal("", 2, 30, 5, SignalType.SINE, duration, precision)
signal1 = Signal("", 5, 10, 0, SignalType.SINE, duration, precision) 
signal2 = Signal("", 1, 80, 0, SignalType.SINE, duration, precision)

composed_signal = signal0.get_function() + signal1.get_function() + signal2.get_function()

N = 300 
F = np.zeros((N, N), dtype=complex)
for i in range(N):
    for j in range(N):
        F[i][j] = math.e**(1j * -2 * np.pi * i * j / N)

X = F.dot(composed_signal)
X = abs(X)
freqs = np.arange(0, N, 1)

cutoff = freqs <= 100
freqs = freqs[cutoff]
X = X[cutoff]

plt.figure()

container = plt.stem(freqs, X)

markerline = container.markerline
stemlines = container.stemlines
baseline = container.baseline

plt.setp(markerline, color='black')
plt.setp(stemlines, color='black')

markerline.set_markerfacecolor('none')

baseline.set_visible(False)

plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(ω)|")
plt.show()
