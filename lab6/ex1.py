import scipy
import numpy as np
import matplotlib.pyplot as plt

B = 3

interval = np.arange(-3, 3, 0.001)
x = np.sinc(B * interval) ** 2

plt.plot(interval, x)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

frequencies = [1, 1.5, 2, 4]

axes = axes.flatten()


for i in range(4):
    T_s = 1 / frequencies[i]

    points = np.arange(-3, 3, 1 / frequencies[i])
    sampled_x = np.sinc(B * points) ** 2

    reconstruced_x = 0.0
    for j in range(len(sampled_x)):
        reconstruced_x += sampled_x[j] * np.sinc((interval - points[j]) / T_s)

    axes[i].set_title(f"F = {frequencies[i]} Hz")
    axes[i].set_xlabel("t[s]")
    axes[i].set_ylabel("Amplitude")
    axes[i].stem(points, sampled_x, linefmt="orange", markerfmt="orange")
    axes[i].plot(interval, x, color="black")
    axes[i].plot(interval, reconstruced_x, color="lime", linestyle="--")

plt.show()


# 1 d)
# cand B este mai mare putem observa un varf mai "ascutit" in jurul punctului 0, in schimb cand B este mai mic
# amplitudinea putem vedea mai degraba un deal.
# de asemenea, cand B este mai mare functia reconstruita nu are aceeasi acuratete pentru frecvente mici