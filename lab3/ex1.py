from ..common import Signal
from ..common import SignalType

import numpy as np
import matplotlib.pyplot as plt
import math

N = 8
F = np.zeros((N, N), dtype=complex)
I = np.identity(N, dtype=complex)

fig, axes = plt.subplots(N, 1, figsize=(10, 12))


for i in range(N):
    for j in range(N):
        F[i][j] = math.e**(1j * -2 * np.pi * i * j / N) 

for i in range(N):
    axes[i].plot(np.arange(0, N, 1), F[i].real)
    axes[i].plot(np.arange(0, N, 1), F[i].imag, linestyle='dashed')

print(np.allclose((np.matmul(F.conj(), F)) / 4, I))
plt.show()
