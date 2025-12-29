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

def fit_ar(data, p, n):
    if n <= p + 1:
        return None, None

    X = [] 
    Y = []
    
    data_subset = data[:n]
    
    for i in range(p, n):
        past_window = data_subset[i-p : i][::-1]
        row = np.insert(past_window, 0, 1.0) # Bias
        Y.append(row)
        X.append(data_subset[i])
        
    X = np.array(X)
    Y = np.array(Y)
    
    beta, residuals, rank, s = np.linalg.lstsq(Y, X, rcond=None)
    predictions = Y.dot(beta)
    
    return beta, predictions

def calculate_mse(data, predictions, n, p):
    actual = data[p:n]
    
    k = min(len(actual), len(predictions))
    
    mse = np.mean((actual[:k] - predictions[:k]) ** 2)
    return mse

p = 100
coefficients, predictions = fit_ar(observed, p, N)

plt.figure(figsize=(12, 6))
plt.plot(X, observed, label="Original", color="gray")
plt.plot(X[p:], predictions, label=f"AR({p}) Prediction", color="blue")
plt.title("AR Model for Time Series")

plt.show()
