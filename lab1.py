import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class SignalType(Enum):
    SINE = 0
    COSINE = 1
    SAWTOOTH = 2
    SQUARE = 3


class Signal:
    def __init__(self, name, freq, phase, signal_type, duration, precision=0.00001):
        self.name = name
        self.freq = freq
        self.phase = phase
        self.signal_type = signal_type
        self.sampling = False
        self.duration = duration
        self.precision = precision

    def with_sampling(self, sampling_freq):
        self.sampling_freq = sampling_freq
        self.sampling = True

        return self

    def plot_signal(self, ax):
        auxiliary_func = lambda x: x
        time_series = np.arange(0, self.duration, self.precision)
        
        if self.sampling:
            sample_time_series = np.arange(0, self.duration, 1/ self.sampling_freq)
        
        if self.signal_type == SignalType.SINE:
            self.phase -= np.pi / 2
        elif self.signal_type == SignalType.SAWTOOTH:
            auxiliary_func = lambda x : x - np.floor(x)
        elif self.signal_type == SignalType.SQUARE:
            auxiliary_func = lambda x: np.sign(x)
        
        if self.signal_type != SignalType.SAWTOOTH:
            f = auxiliary_func(np.cos(2 * np.pi * self.freq * time_series + self.phase))
            if self.sampling:
                sample_f = auxiliary_func(np.cos(2 * np.pi * self.freq * sample_time_series + self.phase))        
        else:
            f = 2 * (time_series * self.freq - np.floor(time_series * self.freq + 1/2))
            if self.sampling:
                sample_f = 2 * (sample_time_series * self.freq - np.floor(sample_time_series * self.freq + 1/2)) 

        if self.sampling:
            ax.stem(sample_time_series, sample_f)

        ax.plot(time_series, f)
        ax.set_title(self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Values")
        ax.grid(True)

#time_series = np.arange(0, 0.03, 0.00005)
#x = Signal("Semnal 1", 520, np.pi * 3, time_series)
#y = Signal("Semnal 2", 280, - np.pi / 3, time_series)
#z = Signal("Semnal 3", 120, + np.pi / 3, time_series)

signals = [Signal("Signal 0", 400, 0, SignalType.SINE, 0.032).with_sampling(5000),
           Signal("Signal 1", 800, 0, SignalType.SINE, 0.03),
           Signal("Signal 2", 240, 0, SignalType.SAWTOOTH, 0.03),
           Signal("Signal 3", 300, 0, SignalType.SQUARE, 0.03)]


fig, axs = plt.subplots(len(signals), 1, figsize=(8,6))

for ax, s in zip(axs, signals):
    s.plot_signal(ax)

plt.tight_layout()
plt.show()

plt.imshow(np.random.rand(128, 128))
plt.show()

mandelbrot = np.zeros((128, 128))

max_iter = 50
x_min, x_max = -2.0, 0.47
y_min, y_max = -1.12, 1.12

for i in range(128):
    for j in range(128):
        x0 = x_min + (x_max - x_min) * j / 128
        y0 = y_min + (y_max - y_min) * i / 128
        x, y = 0.0, 0.0
        iteration = 0
        while (x * x + y * y <= 4) and (iteration < max_iter):
            xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
            iteration += 1
        mandelbrot[i, j] = iteration

plt.imshow(mandelbrot) 
plt.show()


# ex 3
# a) 2000 hz -> 2000 de cicluri pe secunda, esantionare de 2000 de ori pe secunda
# 1 / 2000 s

# b) 3600 s * nr esantionari pe secunda * 4 
# -> 3600 * 2000 * 4 biti -> 28 800 800 biti -> 3 600 000 bytes
# 3 600 000 / (1024^2) = ~ 3.4 MB
