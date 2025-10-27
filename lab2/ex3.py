from common import Signal
from common import SignalType
import sounddevice as sd
from scipy.io import wavfile

sample_rate = 10000
signals = [Signal("Signal 0", 1, 400, 0, SignalType.SINE, 0.32).with_sampling(5000),
           Signal("Signal 1", 1, 800, 0, SignalType.SINE, 0.3),
           Signal("Signal 2", 1, 240, 0, SignalType.SAWTOOTH, 0.3),
           Signal("Signal 3", 1, 300, 0, SignalType.SQUARE, 0.3)]

for signal in signals:
    sd.play(signal.get_function(), sample_rate)
    sd.wait()

    if signal.name == "Signal 0":
        wavfile.write("signal0.wav", sample_rate, signal.get_function())

sample_rate_from_file ,signal_from_file = wavfile.read("signal0.wav")
sd.play(signal_from_file, sample_rate_from_file)

