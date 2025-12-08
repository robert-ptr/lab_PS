import matplotlib.pyplot as plt
import numpy as np

#data
n1_values = np.linspace(-1, 1, 100)
n2_values = np.linspace(-1, 1, 100)
N1, N2 = np.meshgrid(n1_values, n2_values)

# functions
Z1 = np.sin(np.pi * 2 * N1 + np.pi * 3 * N2)
Z2 = np.sin(np.pi * 4 * N1) + np.cos(np.pi * 6 * N2)

fig = plt.figure(figsize=(12, 8)) # Increased size slightly to fit bars

# PLOT 1: Z1 Spatial
ax1 = fig.add_subplot(341)
heatmap1 = ax1.pcolormesh(N1, N2, Z1, cmap='viridis', shading='auto')
fig.colorbar(heatmap1, ax=ax1) # Added colorbar

# PLOT 2: Z1 Frequency
ax2 = fig.add_subplot(342)
Y = np.fft.fft2(Z1)
freq_db = 20 * np.log10(abs(Y) + 1e-9) # Added epsilon to avoid log(0)
im2 = ax2.imshow(freq_db)
fig.colorbar(im2, ax=ax2) # Added colorbar

# PLOT 3: Z2 Spatial
ax3 = fig.add_subplot(343)
heatmap2 = ax3.pcolormesh(N1, N2, Z2, cmap='viridis', shading='auto')
fig.colorbar(heatmap2, ax=ax3) # Added colorbar

# PLOT 4: Z2 Frequency
ax4 = fig.add_subplot(344)
Y = np.fft.fft2(Z2)
Y_shifted = np.fft.fftshift(Y) 
freq_db2 = 20 * np.log10(abs(Y_shifted) + 1e-9)
im4 = ax4.imshow(freq_db2)
fig.colorbar(im4, ax=ax4) # Added colorbar

# PLOT 5: Z3 Frequency 
N = 100
Y = np.zeros((N, N), dtype=complex)
Y[0, 5] = 1
Y[0, N - 5] = 1

Z3 = np.fft.ifft2(Y)

ax5 = fig.add_subplot(345)
im5 = ax5.imshow(abs(Y), cmap='viridis', origin='lower')
fig.colorbar(im5, ax=ax5) # Added colorbar

# PLOT 6: Z3 Spatial
ax6 = fig.add_subplot(346)
im6 = ax6.imshow(np.real(Z3))
fig.colorbar(im6, ax=ax6) # Added colorbar

# PLOT 7: Z4 Frequency
Y = np.zeros((N, N), dtype=complex)
Y[5, 0] = Y[N - 5, 0] = 1 # Corrected assignment logic slightly for numpy

Z4 = np.fft.ifft2(Y)

ax7 = fig.add_subplot(347)
im7 = ax7.imshow(abs(Y), cmap='viridis', origin='lower')
fig.colorbar(im7, ax=ax7) # Added colorbar

# PLOT 8: Z4 Spatial
ax8 = fig.add_subplot(348)
im8 = ax8.imshow(np.real(Z4))
fig.colorbar(im8, ax=ax8) # Added colorbar

# PLOT 9: Z5 Frequency
Y = np.zeros((N, N), dtype=complex)
Y[5, 5] = Y[N - 5, N - 5] = 1

Z5 = np.fft.ifft2(Y)

ax9 = fig.add_subplot(349)
im9 = ax9.imshow(abs(Y), cmap='viridis')
fig.colorbar(im9, ax=ax9)

# PLOT 10: Z5 Spatial
ax10 = fig.add_subplot(3,4,10)
im10 = ax10.imshow(np.real(Z5))
fig.colorbar(im10, ax=ax10)

plt.tight_layout()
plt.show()
