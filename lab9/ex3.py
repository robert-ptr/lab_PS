import numpy as np
import matplotlib.pyplot as plt

N = 1000

X = np.linspace(0, 10, N)

trend = X ** 2 - 4 * X + 7
seasonal = np.sin(2 * np.pi * X * 5) + np.sin(2 * np.pi * X * 30)
residuals = np.random.normal(0, 2, size=N)
observed = trend + seasonal + residuals

q = 10

errors = []
averages = []

for i in range(q - 1, len(observed)):
    window = observed[i - q + 1 : i + 1]
    ma = np.mean(window)

    y_actual = observed[i]
    error = y_actual - ma

    averages.append(ma)
    errors.append(error)

    print(f"{i} | {y_actual:.2f} | {str(window)} | {ma:.2f} | {error:.2f}")

plt.plot(X[q-1:], observed[q-1:], label="Original")
plt.plot(X[q-1:], averages, label="Moving Average")

plt.title("Smoothing Time Series with Moving Average")
plt.legend()

plt.show()
