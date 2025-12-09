from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = datasets.face(gray=True)

def calculate_snr(original, noisy_version):
    noise = original - noisy_version
    norm_orig = np.linalg.norm(original)
    norm_noise = np.linalg.norm(noise)
    if norm_noise == 0: return 100 
    return 20 * np.log10(norm_orig / norm_noise)

pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=X.shape)

X_noisy = X.astype(float) + noise
X_noisy = np.clip(X_noisy, 0, 255)

snr_noisy = calculate_snr(X, X_noisy)
print(f"{snr_noisy:.2f} dB")

Y_noisy = np.fft.fft2(X_noisy)
Y_noisy_shifted = np.fft.fftshift(Y_noisy)

rows, cols = X.shape
crow, ccol = rows // 2, cols // 2

R = 70 

mask = np.zeros((rows, cols))
y, x = np.ogrid[:rows, :cols]
mask_area = (x - ccol)**2 + (y - crow)**2 <= R**2
mask[mask_area] = 1

Y_cleaned = Y_noisy_shifted * mask

X_cleaned = np.fft.ifft2(np.fft.ifftshift(Y_cleaned))
X_cleaned = np.real(X_cleaned)

snr_cleaned = calculate_snr(X, X_cleaned)
print(f"{snr_cleaned:.2f} dB")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(X, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')

ax[1].imshow(X_noisy, cmap='gray')
ax[1].set_title(f"Noisy (SNR={snr_noisy:.2f} dB)")
ax[1].axis('off')

ax[2].imshow(X_cleaned, cmap='gray')
ax[2].set_title(f"Filtered (SNR={snr_cleaned:.2f} dB)")
ax[2].axis('off')

plt.show()
