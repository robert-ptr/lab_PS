import numpy as np  
import scipy

#x = np.random.randint(low=0, high=100, size=20)
t = np.arange(20)
x = np.sin(2*np.pi*t/20)

d = 5

y = np.roll(x, d)

d2 = scipy.fft.ifft(scipy.fft.fft(x) * scipy.fft.fft(y))
d3 = scipy.fft.ifft(scipy.fft.fft(y) / scipy.fft.fft(x))

print(d2)
print(d3)

idx = np.argmax(d2.real)
print(idx)