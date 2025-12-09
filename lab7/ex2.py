import matplotlib.pyplot as plt
import numpy as np
from scipy import datasets

X = datasets.face(gray=True)
rows, cols = X.shape
crow, ccol = rows // 2, cols // 2

Y = np.fft.fft2(X)
Y_shifted = np.fft.fftshift(Y)

def calculate_snr(original, compressed):
    noise = original - compressed
    norm_orig = np.linalg.norm(original)
    norm_noise = np.linalg.norm(noise)
    if norm_noise == 0: return 100 
    return 20 * np.log10(norm_orig / norm_noise)

target_snr = 18

best_R = 0
found_snr = 0
X_final = X.copy()

for R in range(min(rows, cols)//2, 1, -2):
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 <= R**2
    
    Y_filtered = Y_shifted * mask
    
    X_rec = np.real(np.fft.ifft2(np.fft.ifftshift(Y_filtered)))
    
    current_snr = calculate_snr(X, X_rec)
    
    if current_snr < target_snr:
        break
    
    best_R = R
    found_snr = current_snr
    X_final = X_rec

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(X, cmap='gray')

ax[1].imshow(X_final, cmap='gray')

plt.show()
