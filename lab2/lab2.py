combined_signal = signals[1].get_function() + signals[2].get_function()
   
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

time_series = np.arange(0, 0.3, 0.001)

# Graficul 1: Signal 1
axes[0].plot(time_series, signals[1].get_function())
axes[0].set_title(signals[1].name) # Folosiți un titlu relevant
axes[0].set_ylabel("Valori")
axes[0].grid(True)

# Graficul 2: Signal 2
axes[1].plot(time_series, signals[2].get_function())
axes[1].set_title(signals[2].name)
axes[1].set_ylabel("Valori")
axes[1].grid(True)

# Graficul 3: Semnalul Combinat
axes[2].plot(time_series, combined_signal)
axes[2].set_title("Combined Signal (Signal 1 + Signal 2)")
axes[2].set_ylabel("Valori")
axes[2].grid(True)

# Setați eticheta X doar pentru ultimul grafic
axes[2].set_xlabel("Timp [s]")
# ex 5

# ex 6

# ex 7

# ex 8

plt.tight_layout()
plt.show()
