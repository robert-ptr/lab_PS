from ..common import Signal
from ..common import SignalType
import sounddevice as sd
import numpy as np

precision = 0.0001

signal1 = Signal("Signal 0", 1, 800, 0, SignalType.SINE, 1, precision)
signal2 = Signal("Signal 1", 1, 300, 0, SignalType.SINE, 1, precision)
combined_signals = np.concatenate((signal1.get_function(), signal2.get_function()))
sd.play(combined_signals, 1 / precision)
sd.wait()

# in urma redarii se observa un sunet cu frecventa inalta urmat brusc de un sunet de frecventa joasa 
