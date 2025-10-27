from common import Signal
from common import SignalType
import matplotlib.pyplot as plt

freq_sample = 800

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

signal0 = Signal("Signal 0", 1, freq_sample / 2, 0, SignalType.SINE, 0.01).with_sampling(freq_sample)
signal1 = Signal("Signal 1", 1, freq_sample / 4, 0, SignalType.SINE, 0.01).with_sampling(freq_sample)
signal2 = Signal("Signal 2", 1, 0, 0, SignalType.SINE, 0.01).with_sampling(freq_sample)

signal0.plot_signal(axes[0])
signal1.plot_signal(axes[1])
signal2.plot_signal(axes[2])

plt.tight_layout()
plt.show()

# signal0 si signal2 "arata" la fel, esantionarea are loc in asa fel incat valorile sunt mereu la fel, deci semnalul pare care are
# frecventa zero
