import numpy as np
import time

N = 3

p = np.random.randint(low=0, high=10, size=N)
q = np.random.randint(low=0, high=10, size=N)

print(q, p)

def basic_polynomial_mul(a, b):
    n = len(a)
    m = len(b)

    result = np.zeros(n + m - 1)

    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]

    return result

def fft_polynomial_mul(a, b):
    n = len(a)
    m = len(b)

    size = n + m - 1
    M = 1 << (size - 1).bit_length()

    A = np.fft.fft(a, M)
    B = np.fft.fft(b, M)

    C = A * B
    c = np.fft.ifft(C)

    return np.rint(c.real[:size]).astype(int).tolist()

start = time.time()

print(basic_polynomial_mul(q, p))

end = time.time()
print("Elapsed:", end - start, "seconds")

start = time.time()

print(fft_polynomial_mul(q, p))

end = time.time()
print("Elapsed:", end - start, "seconds")