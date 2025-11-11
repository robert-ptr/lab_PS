from ..common import SignalType
from ..common import Signal
import numpy as np
import matplotlib.pyplot as plt

duration = 0.05 # seconds

signal0 = Signal("Signal 0", amplitude=1, freq=200, phase=0, signal_type=SignalType.SINE, duration=duration).with_sampling(400)
signal1 = Signal("Signal 1", amplitude=1, freq=100, phase=0, signal_type=SignalType.SINE, duration=duration).with_sampling(400)
signal2 = Signal("Signal 2", amplitude=1, freq=40, phase=0, signal_type=SignalType.SINE, duration=duration).with_sampling(400)

fig, axes = plt.subplots(3, figsize=(10, 6))

signal0.plot_signal(axes[0])
signal1.plot_signal(axes[1])
signal2.plot_signal(axes[2])

plt.show()

# Frecventa de esantionare trebuie sa fie cel putin dublu fata de frecventa semnalului pentru a evita fenomenul de aliere.
# Este suficient sa folosim o frecventa de esantionare de 400 Hz, deoarece cea mai mare frecventa a semnalelor este de 200 Hz.
# In acest caz, toate cele 3 semnale sunt esantionate corect, fara a aparea fenomenul de aliere.