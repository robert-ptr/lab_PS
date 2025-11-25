from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

data_path = Path(__file__).resolve().parent.parent / "lab6" / "Train.csv"
x = np.genfromtxt(data_path, delimiter=',')
x = x[1:, 2]
x = x[:72]

weights = [5, 9, 13, 17]

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(np.arange(0, 72, 1), x)
axes[0].set_title("Original")
axes[1].set_title("Smoothed Out")
for w in weights:
    axes[1].plot(np.arange(0, 72 - w + 1, 1), np.convolve(x, np.ones(w), 'valid') / w)

plt.show()

# 6 c)
# frecventa de esantionare = 1 / 3600 Hz
# frecventa nquist este jumatate din frecventa de esantionare, deci = 1 / 7200 Hz
# frecventa aleasa este 1 / 14400 Hz(valoarea normalizata este 0.5)
# , ceea ce inseamna o esantionare din 4 in 4 ore
# in felul acesta scapam de blocaje din trafic, sau rezultatele datorate unui senzor defect
# dar putem observa variatiile de pe parcursul unei zile, datorate, de exemplu, orelor de varf
# cand majoritatea oamenilor merg la munca

# 6 d)
Wn = 0.5
N = 5
rp = 5

den_butt, nom_butt = signal.butter(N, Wn, btype='low')
den_chebyshev, nom_chebyshev = signal.cheby1(N, rp, Wn, btype='low')

x_butt = signal.filtfilt(den_butt, nom_butt, x)
x_cheby = signal.filtfilt(den_chebyshev, nom_chebyshev, x)

# 6 e)
plt.figure(figsize=(12, 6))
plt.plot(x, 'lightgray', label='Original Data', linewidth=2)
plt.plot(x_butt, 'b-', label='Butterworth Filter (Order 5)')
plt.plot(x_cheby, 'r--', label=f'Chebyshev Filter (rp={rp}dB)')

plt.title('Butterworth vs Chebyshev')
plt.xlabel('Time (hour)')
plt.ylabel('N Vehicles')
plt.legend()
plt.grid(True)
plt.show()

# as alege fltrul Butterworth deoarece mi se pare ca evenimentele periodice sunt mai usor de observat
# in comparatie cu filtrul Chebyshev

# 6 f)
plt.plot(x, 'k', alpha=0.3, label='Original', linewidth=3)

orders = [2, 5, 9]

for N in orders:
    b, a = signal.butter(N, Wn, btype='low')
    x_filt = signal.filtfilt(b, a, x)
    plt.plot(x_filt, linewidth=2, label=f'Butterworth Order {N}')

plt.title('Butterworth')
plt.legend()
plt.grid(True)

rps = [0.1, 1, 5]

fig, axs = plt.subplots(len(rps), len(orders), figsize=(15, 12), sharex=True, sharey=True)

for i, rp in enumerate(rps):
    for j, N in enumerate(orders):
        b, a = signal.cheby1(N, rp, Wn, btype='low')
        x_filt = signal.filtfilt(b, a, x)
        
        ax = axs[i, j]
        ax.plot(x, 'k', alpha=0.3, linewidth=1.5)
        ax.plot(x_filt, 'r', linewidth=2)
        
        ax.grid(True, alpha=0.5)
        
        if j == 0:
            ax.set_ylabel(f'rp={rp} dB\n')
        
        if i == 0:
            ax.set_title(f'Order N={N}')

plt.tight_layout()
plt.show()