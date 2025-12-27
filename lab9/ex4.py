import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

N = 1000

X = np.linspace(0, 10, N)

trend = X ** 2 - 4 * X + 7
seasonal = np.sin(2 * np.pi * X * 5) + np.sin(2 * np.pi * X * 30)
residuals = np.random.normal(0, 2, size=N)
observed = trend + seasonal + residuals

fig = plt.figure(figsize=(12, 10))

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

ax1 = fig.add_subplot(221)
ax1.plot(X[q-1:], observed[q-1:], label="Original", color="gray")
ax1.plot(X[q-1:], averages, label=f"MA({q})", color="blue")
ax1.set_title("MA Model for Time Series")
ax1.legend()

def fit_ar(data, p, n):
    if n <= p + 1:
        return None, None

    X_mat = [] 
    Y_vec = []
    
    data_subset = data[:n]
    
    for i in range(p, n):
        past_window = data_subset[i-p : i][::-1]
        row = np.insert(past_window, 0, 1.0)
        Y_vec.append(row)
        X_mat.append(data_subset[i])
        
    X_mat = np.array(X_mat)
    Y_vec = np.array(Y_vec)
    
    beta, residuals, rank, s = np.linalg.lstsq(Y_vec, X_mat, rcond=None)
    predictions = Y_vec.dot(beta)
    
    return beta, predictions

def calculate_mse(data, predictions, n, p):
    actual = data[p:n]
    k = min(len(actual), len(predictions))
    mse = np.mean((actual[:k] - predictions[:k]) ** 2)
    return mse

p_initial = 100
coefficients, predictions_initial = fit_ar(observed, p_initial, N)

ax2 = fig.add_subplot(222)
ax2.plot(X, observed, label="Original", color="gray")
ax2.plot(X[p_initial:], predictions_initial, label=f"AR({p_initial}) Prediction", color="blue")
ax2.set_title("AR Model for Time Series")
ax2.legend()

split_idx = int(N * 0.8)
train_data = observed[:split_idx]
test_data = observed[split_idx:]

best_mse = float('inf')
best_order = (0, 0, 0)

for p_test in range(1, 21, 5): 
    for q_test in range(1, 21, 5):
        try:
            model = ARIMA(train_data, order=(p_test, 0, q_test))
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=len(test_data))
            
            current_mse = calculate_mse(test_data, forecast, len(test_data), 0)
            
            if current_mse < best_mse:
                best_mse = current_mse
                best_order = (p_test, 0, q_test)
        except:
            continue

p = best_order[0]
q = best_order[2]

_, ar_predictions_optimal = fit_ar(observed, p, N)

start = max(p, q)
sliced_ar_predictions = ar_predictions_optimal[start - p:]
sliced_observed = observed[start:]

ar_errors = sliced_observed - sliced_ar_predictions

ar_correction = []
for i in range(len(ar_errors)):
    if i < q:
        ar_correction.append(0)
    else:
        error_window = ar_errors[i - q : i]
        ar_correction.append(np.mean(error_window))

arma_prediction_manual = sliced_ar_predictions + ar_correction

ax3 = fig.add_subplot(223)
ax3.plot(X[start:], sliced_observed, label="Original", color="gray")
ax3.plot(X[start:], arma_prediction_manual, label=f"ARMA({p}, {q}) Manual", color="blue")
ax3.set_title(f"ARMA Model for Time Series")
ax3.legend()

model = ARIMA(observed, order=best_order) 
model_fit = model.fit()
stats_predictions = model_fit.predict(start=0, end=N-1)

p_auto, d_auto, q_auto = model_fit.model.order
burn_in = max(p_auto, q_auto)

ax4 = fig.add_subplot(224)
ax4.plot(X[burn_in:], observed[burn_in:], label="Original", color="gray")
ax4.plot(X[burn_in:], stats_predictions[burn_in:], label=f"ARMA({p_auto}, {q_auto}) Statsmodels", color="blue")
ax4.set_title("ARMA (statsmodels) Model for Time Series")
ax4.legend()

plt.tight_layout()
plt.show()