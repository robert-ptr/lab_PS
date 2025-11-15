import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from scipy.signal import find_peaks

# 1 a) frecventa de esantionare a semnalului este 1 / 3600 Hz
# 1 b) 18287 de ore = 762 zile (aproximativ)
# 1 c)
data_path = Path(__file__).resolve().parent.parent / "lab5" / "Train.csv"
x = np.genfromtxt(data_path, delimiter=',')
data = x[1:, 2]
N = len(data)

X = np.fft.fft(data)
X = abs(X / N)
X = X[:N//2]

Fs = 1 / 3600
f = Fs * np.linspace(0, N/2, N//2) / N

cutoff = 0.1
X_without_dc = X[10:]
freq_without_dc = f[10:]

A_max = X_without_dc.max()
mask = X_without_dc > cutoff * A_max

freq_max = freq_without_dc[mask].max()
print("Freq max (Hz):", freq_max)

# 1 d)

plt.figure(figsize=(10, 5))
plt.plot(freq_without_dc, X_without_dc)
plt.xlabel("Freq (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# 1 e)
# prezinta o componenta continua, pe care am eliminat-o anterior

# 1 f)
peaks, _ = find_peaks(X_without_dc, height = np.max(X_without_dc) * 0.1)
peak_amps = X_without_dc[peaks]
top4_sorted_indices = np.argsort(peak_amps)[-4:]
top4_peak_indices = peaks[top4_sorted_indices]
top4_frequencies = freq_without_dc[top4_peak_indices]

print("Top 4 frequencies:")
print(top4_frequencies)

# frecventele gasite sunt: [3.31158020e-06 2.31506799e-05 1.65579010e-06 1.15753400e-05]
# daca facem niste calculate, reiese ca ele corespund unor perioade(in ordinea lor din array)
# de 3.5 zile, 12 ore, 7 zile, 24 de ore
# cele 12 ore banuiesc ca corespund orei de varf de dimineata si orei de varf de dupa-amiaza
# cele 24 reprezinta ciclul zilnic al activitatii umane
# cele 7 zile reprezinta rutina saptamanala
# cel mai ciudat rezultat este cel de 3.5 zile, nu stiu sigur ce inseamna. O diferenta intre luni-joi si joi-duminica?

# 1 g)

# 1 h)
# In primul rand nu cred ca putem afla anul. Am putea afla deceniul insa asta depinde si de locatie. Deoarece in unele zone
# traficul a crescut pe parcursul timpului, iar in altele(cum ar fi orasele care au inceput sa inchida accesul masinilor in centru) a scazut.

# De fapt, probabil am putea afla anul daca masuratoarea a inceput in perioada pandemiei dar include si perioade de trafic normal(fie inainte sau dupa pandemie, ceea ce in cazul nostru ar trebui sa fie valabil deoarece datele acopera 2 ani).

# Am putea afla luna pe baza perioadelor in care oamenii isi iau vacanta. Daca nu este o locatie turistica, probabil vom vedea mai putin trafic
# de Craciun, Paste, vara. Deci, daca le indentificam, putem in functie de ele sa stabilim luna in care a inceput masuratoarea
# Pe urma putem afla ziua in care a avut loc cea mai apropiata sarbatoare si putem observa la cate zile distanta se afla de ziua in care
# a inceput masuratoarea.
# Si in cazul acesta suntem dependenti de locatie, nu toate tarile au aceleasi sarbatori.
# De asemenea, putem avea evenimente "lebada neagra" care sa ne induca in eroare. Am putea observa trafic redus si sa presupunem ca a fost
# o sarbatoare, cand de fapt a nins prea mult.

# In concluzie, este posibil sa aflam ziua si luna, poate chiar si anul. Insa cred ca sunt sanse mari sa existe mai multe scenarii posibile.
# Prin asta vreau sa spun ca putem avea mai multe date care par rezonabile, depinzand de locatie
# caz in care cred ca ne ramane doar sa o luam probabilistic, in sensul in care putem sa ne gandim in ce zone de pe glob este
# cel mai probabil ca traficul sa fie monitorizat, sa fie atat de mare, in ce tari se circula mai mult cu masina, etc.

# 1 i)var

sample_spacing = 3600 

X_full = np.fft.fft(data)

freqs = np.fft.fftfreq(N, d=sample_spacing)

max_frequency = 3.4e-6 # elimina fluctuatiile de pe parcursul unei zile 
mask = np.abs(freqs) < max_frequency

X_filtered = X_full * mask

filtered_signal_fft = np.real(np.fft.ifft(X_filtered))

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(data, label="Semnal Original", alpha=0.7)
axes[1].plot(filtered_signal_fft, label=f"Filtrat (taiere < {max_frequency} Hz)", linewidth=2, color='red')
plt.show()
