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

# MA 
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

ax1 = fig.add_subplot(221)
ax1.plot(X[q-1:], observed[q-1:], label="Original", color="gray")
ax1.plot(X[q-1:], averages, label=f"MA({q})", color="blue")

ax1.set_title("MA Model for Time Series")
ax1.legend()

# AR
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

ax2 = fig.add_subplot(222)

ax2.plot(X, observed, label="Original", color="gray")
ax2.plot(X[p:], predictions, label=f"AR({p}) Prediction", color="blue")
ax2.set_title("AR Model for Time Series")
ax2.legend()

# ARMA



# calculate optimal p, q values
for i in range(1, 20):
    for j in range(1, 20):
        pass

start = max(p, q)
predictions = predictions[start - p:]
sliced_observed = observed[start:]

ar_errors = sliced_observed - predictions

ar_correction = []
for i in range(len(ar_errors)):
    if i < q:
        ar_correction.append(0)
    else:
        error = ar_errors[i - q : i]
        ar_correction.append(np.mean(error))

arma_prediction = predictions + ar_correction

ax3 = fig.add_subplot(223)
ax3.plot(X[start:], sliced_observed, label="Original", color="gray")
ax3.plot(X[start:], arma_prediction, label=f"ARMA({p} {q}) Prediction", color="blue")
ax3.set_title("ARMA Model for Time Series")
ax3.legend()

# ARMA using statsmodel

model = ARIMA(observed, order=(1,1,1))
model_fit = model.fit()
stats_predictions = model_fit.predict(start=0, end=N-1)

p_auto, d_auto, q_auto = model_fit.model.order

burn_in = max(p_auto, q_auto)

ax4 = fig.add_subplot(224)
ax4.plot(X[burn_in:], observed[burn_in:], label="Original", color="gray")
ax4.plot(X[burn_in:], stats_predictions[burn_in:], label="ARMA{p_auto} {q_auto} Preiction", color="blue")
ax4.set_title("ARMA (statsmodels) Model for Time Series")
ax4.legend()

plt.show()
