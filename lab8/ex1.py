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

# 1 d
# cel mai bun rezultat: p = 841, m = 921, MSE = 2.7473

def evaluate_test_set(data, beta, p, split_index, test_size):
    errors = []
    end_index = min(split_index + test_size, len(data))
    
    for k in range(split_index, end_index):
        history = data[k-p : k][::-1]
        row = np.insert(history, 0, 1.0)
        prediction = np.dot(row, beta)
        actual = data[k]
        errors.append((actual - prediction) ** 2)
        
    if len(errors) == 0:
        return float('inf')
        
    return np.mean(errors)

best_p = 0
best_m = 0
best_error = float('inf')
test_size = 50

for i in range(1, 1000, 10):
    for j in range(i + 20, N - test_size, 10):
        
        beta, _ = fit_ar(observed, i, j)
        
        if beta is None:
            continue
            
        error = evaluate_test_set(observed, beta, i, j, test_size)

        if error < best_error:
            best_error = error
            best_p = i
            best_m = j
            print(f"New Best -> p: {best_p}, m: {best_m}, MSE: {best_error:.4f}")

print(f"FINAL RESULT: Best p: {best_p}, m: {best_m}, MSE: {best_error:.4f}")
