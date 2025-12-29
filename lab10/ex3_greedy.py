import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit

N = 1000

X = np.linspace(0, 10, N)

trend = X ** 2 - 4 * X + 7
seasonal = np.sin(2 * np.pi * X * 5) + np.sin(2 * np.pi * X * 30)
residuals = np.random.normal(0, 2, size=N)
observed = trend + seasonal + residuals

def fit_ar_greedy(data, p, n, n_nonzero_coefs=10):
    if n <= p + 1:
        return None, None

    targets = [] 
    features = []
    
    data_subset = data[:n]
    
    for i in range(p, n):
        past_window = data_subset[i-p : i][::-1]
        row = np.insert(past_window, 0, 1.0) # Bias
        features.append(row)
        targets.append(data_subset[i])
        
    targets = np.array(targets)
    features = np.array(features)
    
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False)
    model.fit(features, targets)
    
    beta = model.coef_
    predictions = model.predict(features)

    return beta, predictions

p = 100

beta_lasso, predictions = fit_ar_greedy(observed, p, N)

zero_count = np.sum(beta_lasso == 0)
print(f"Total coefficients: {len(beta_lasso)}")
print(f"Coefficients set to ZERO by Lasso: {zero_count}")

plt.figure(figsize=(12, 6))
plt.plot(X, observed, label="Original", color="gray")
plt.plot(X[p:], predictions, label=f"AR({p}) Prediction", color="blue")
plt.title("AR Model for Time Series")

plt.show()
