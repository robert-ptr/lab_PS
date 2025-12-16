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

def calculate_mse(data, predictions, n, p):
    actual = data[p:n]
    
    k = min(len(actual), len(predictions))
    
    mse = np.mean((actual[:k] - predictions[:k]) ** 2)
    return mse

# 1 b 
alpha_values = np.linspace(0, 1, 50)
beta_values = np.linspace(0, 1, 50)
gamma_values = np.linspace(0, 1, 50)
best_error = 10000
best_alpha = 0
best_beta = 0
best_gamma = 0

s = np.zeros(N)
b = np.zeros(N)
c = np.zeros(N)
prediction = np.zeros(N)

L = 20

for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                s[L - 1] = observed[L - 1]
                b[L - 1] = (observed[L - 1] - observed[0]) / L
                c[:L] = 0
                prediction[0] = observed[0] 
                for i in range(L, N):
                    s[i] = alpha * (observed[i] - c[i - L]) + (1 - alpha) * (s[i - 1] + b[i - 1])
                    b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
                    c[i] = gamma * (observed[i] - s[i] - b[i - 1]) + (1 - gamma) * c[i - L]
                    prediction[i] = s[i - 1] + b[i - 1] + c [i - L]

                error = calculate_mse(observed, prediction[1:], N, 1)
                if error < best_error:
                    best_error = error
                    best_beta = beta
                    best_alpha = alpha
                    best_gamma = gamma

prediction = np.zeros(N)
s = np.zeros(N)
b = np.zeros(N)
c = np.zeros(N)
s[L - 1] = observed[L - 1]
b[L - 1] = (observed[L - 1] - observed[0]) / L
c[:L] = 0

for i in range(L, N):
    s[i] = best_alpha * (observed[i] - c[i - L]) + (1 - best_alpha) * (s[i - 1] + b[i - 1])
    b[i] = best_beta * (s[i] - s[i - 1]) + (1 - best_beta) * b[i - 1]
    c[i] = best_gamma * (observed[i] - s[i] - b[i - 1]) + (1 - best_gamma) * c[i - L]
    prediction[i] = s[i - 1] + b[i - 1] + c[i - L]

plt.plot(X, observed, color="grey")
plt.plot(X, prediction, color="red")
plt.title("Third Order Exponential Smoothing")
plt.show()

print(f"Best alpha: {best_alpha}, Best beta: {best_beta}, Best gamma: {best_gamma}, MSE: {best_error}")
