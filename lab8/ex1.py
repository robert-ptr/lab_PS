import numpy as np
import scipy
import matplotlib.pyplot as plt

# 1 a

N = 1000

X = np.linspace(0, 10, N)
print(X)
trend = X ** 2 - 4 * X + 7
seasonal = np.sin(2 * np.pi * X * 5) + np.sin(2 * np.pi * X * 30)
residuals = np.random.normal(0, 2, size=N)

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(X, trend)

ax2 = fig.add_subplot(222)
ax2.plot(X, seasonal)

ax3 = fig.add_subplot(223)
ax3.plot(X, residuals)

ax4 = fig.add_subplot(224)
ax4.plot(X, trend + seasonal + residuals)

plt.show()
