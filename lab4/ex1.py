from ..common import Signal
from ..common import SignalType
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def fft(X, invert=False):
    N = len(X)
    if N <= 1:
        return X
    even = fft(X[0::2], invert)
    odd = fft(X[1::2], invert)

    ang = 2 * np.pi / N * (-1 if invert else 1)
    w = complex(1)
    wn = complex(np.cos(ang), np.sin(ang))

    for k in range(N // 2):
        X[k] = even[k] + w * odd[k]
        X[k + N // 2] = even[k] - w * odd[k]
        if invert:
            X[k] /= 2
            X[k + N // 2] /= 2
        w *= wn

    return X

N_values = [128, 256, 512, 1024, 2048, 4096, 8192]
custom_durations = []
numpy_durations = []

for value in N_values:
    custom_fft = [complex(i) for i in np.random.rand(value)]
    numpy_fft = [complex(i) for i in np.random.rand(value)]

    start_time = time.time()
    fft(custom_fft)
    custom_duration = time.time() - start_time
    custom_durations.append(custom_duration)

    start_time = time.time()
    np.fft.fft(numpy_fft)
    numpy_duration = time.time() - start_time
    numpy_durations.append(numpy_duration)

    print(f"N = {value}, custom fft time: {custom_duration:.6f}s, numpy fft time: {numpy_duration:.6f}s")

plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Time (s)')
plt.plot(N_values, custom_durations, label='Custom FFT', marker='o')
plt.plot(N_values, numpy_durations, label='NumPy FFT', marker='o')
plt.show()