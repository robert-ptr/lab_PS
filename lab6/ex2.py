import numpy as np
import matplotlib.pyplot as plt

vec = np.random.rand(100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(4):
    axes[i].set_title(f"Iteration {i}")
    axes[i].plot(np.arange(0, 100, 1), vec)
    vec = vec * vec

plt.show()

vec = np.concatenate([
    np.zeros(20, dtype=int),
    np.ones(20, dtype=int),
    np.zeros(60, dtype=int)
])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(4):
    axes[i].set_title(f"Iteration {i}")
    axes[i].plot(np.arange(0, 100, 1), vec)
    vec = vec * vec

plt.show()

# in cazul unui vector aleator, cu atat iteram mai mult cu atat vedem mai putine spike-uri si mai multe zone cu valoarea 0
# in cazul unui semnal bloc rectangular, nimic nu se schimba
