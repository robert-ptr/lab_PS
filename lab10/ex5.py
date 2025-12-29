import numpy as np
from sklearn.linear_model import Lasso

N = 1000

X = np.linspace(0, 10, N)
trend = X ** 2 - 4 * X + 7
seasonal = np.sin(2 * np.pi * X * 5) + np.sin(2 * np.pi * X * 30)
residuals = np.random.normal(0, 2, size=N)
observed = trend + seasonal + residuals

def fit_ar_lasso(data, p, n, alpha=0.1):
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
    
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000) # regularizare L1
                                                                    # inainte foloseam lstsq 
    model.fit(features, targets)
    
    beta = model.coef_
    predictions = model.predict(features)

    return beta, predictions

p = 100
beta_lasso, predictions = fit_ar_lasso(observed, p, N, 0.3)

def get_polynomial_roots(coeffs): # doesn't include the dominant coefficient(that is 1)
    coeffs=  np.array(coeffs, dtype=float)

    n = len(coeffs)

    C = np.zeros((n, n))

    idx = np.arange(n - 1)
    C[idx + 1, idx] = 1
    C[:, -1] = -coeffs[::-1]
    print(C)
    roots = np.linalg.eigvals(C)

    return roots

roots = get_polynomial_roots(-beta_lasso[1:])
print(roots)

max_root = np.max(np.abs(roots))

if max_root < 1:
    print("Model is stationary")
else:
    print("Model is NOT stationary")

