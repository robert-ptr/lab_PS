from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

class SignalType(Enum):
    SINE = 0
    COSINE = 1
    SAWTOOTH = 2
    SQUARE = 3

class Signal:
    def __init__(self, name, amplitude, freq, phase, signal_type, duration, precision=0.00001):
        self.name = name
        self.amplitude = amplitude
        self.freq = freq
        self.phase = phase
        self.signal_type = signal_type
        self.sampling = False
        self.duration = duration
        self.precision = precision

        auxiliary_func = lambda x: x

        time_series = np.arange(0, self.duration, self.precision)
        phase = self.phase
        freq = self.freq

        if self.signal_type == SignalType.SINE:
            phase -= np.pi / 2
        elif self.signal_type == SignalType.SAWTOOTH:
            auxiliary_func = lambda x : x - np.floor(x)
        elif self.signal_type == SignalType.SQUARE:
            auxiliary_func = lambda x: np.sign(x)
        
        if self.signal_type != SignalType.SAWTOOTH:
            self.f = auxiliary_func(self.amplitude * np.cos(2 * np.pi * freq * time_series + phase))
        else:
            self.f = self.amplitude * 2 * (time_series * freq - np.floor(time_series * freq + 1/2))

    def with_sampling(self, sampling_freq):
        self.sampling_freq = sampling_freq
        self.sampling = True

        return self

    def get_function(self):
        return self.f

    def plot_signal(self, ax):
        time_series = np.arange(0, self.duration, self.precision)
        phase = self.phase
        auxiliary_func = lambda x: x

        if self.sampling:
            sample_time_series = np.arange(0, self.duration, 1 / self.sampling_freq)
        
        if self.signal_type == SignalType.SINE:
            phase -= np.pi / 2
        elif self.signal_type == SignalType.SAWTOOTH:
            auxiliary_func = lambda x : x - np.floor(x)
        elif self.signal_type == SignalType.SQUARE:
            auxiliary_func = lambda x: np.sign(x)


        if self.signal_type != SignalType.SAWTOOTH and self.sampling:
            sample_f = auxiliary_func(self.amplitude * np.cos(2 * np.pi * self.freq * sample_time_series + phase))        
        elif self.sampling:
                sample_f = self.amplitude * 2 * (sample_time_series * freq - np.floor(sample_time_series * self.freq + 1/2)) 
        
        if self.sampling:
            ax.stem(sample_time_series, sample_f)

        ax.plot(time_series, self.f)
        ax.set_title(self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Values")
        ax.grid(True)
