import numpy as np
import scipy
import matplotlib.pyplot as plt

# 1 a

N = 1000

X = np.linspace(0, 10, N)

trend = X ** 2 - 4 * X + 7
seasonal = np.sin(2 * np.pi * X * 5) + np.sin(2 * np.pi * X * 30)
residuals = np.random.normal(0, 2, size=N)
observed = trend + seasonal + residuals

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(X, trend)
ax1.set_title("Trend")

ax2 = fig.add_subplot(222)
ax2.plot(X, seasonal)
ax2.set_title("Seasonal")

ax3 = fig.add_subplot(223)
ax3.plot(X, residuals)
ax3.set_title("Residuals")

ax4 = fig.add_subplot(224)
ax4.plot(X, observed)
ax4.set_title("Observed")

plt.show()

# 1 b

observed_centered = observed - np.mean(observed)
correlation = np.correlate(observed_centered, observed_centered, "full")
correlation = correlation[correlation.size // 2:] / correlation[0] 
plt.plot(X, correlation)
plt.title("Correlation")
plt.show()

# 1 c

def fit_ar_numpy(data, p):
    n = len(data)
    X = []
    Y = []
    
    for i in range(p, n):
        past_window = data[i-p : i][::-1]
        row = np.insert(past_window, 0, 1.0)
        Y.append(row)
        X.append(data[i])
        
    X = np.array(X)
    Y = np.array(Y)
    
    beta, residuals, rank, s = np.linalg.lstsq(Y, X)
    predictions = Y.dot(beta)
    
    return beta, predictions

p = 100
coefficients, predictions = fit_ar_numpy(observed, p)

plt.figure(figsize=(12, 6))
plt.plot(X, observed, label='Original', color='gray')
plt.plot(X[p:], predictions, label=f'AR({p}) Prediction', color='blue')
plt.title("AR Model for Time Series")

plt.show()
