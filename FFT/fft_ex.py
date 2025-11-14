import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 1e3                       # معدل العينة 1000 Hz
t = np.arange(0, 1, 1/fs)      # 1 ثانية

# Generate signal
signal = np.sin(2*np.pi*50*t)  # موجة بتردد 50 Hz

# FFT
fft_values = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/fs)

# Magnitude & Phase
magnitude = np.abs(fft_values)
phase = np.angle(fft_values)

# -----------------------------
# 1) Time Domain Plot
# -----------------------------
plt.figure()
plt.plot(t, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

# -----------------------------
# 2) FFT Magnitude Plot
# -----------------------------
plt.figure()
plt.plot(freqs, magnitude)
plt.title("Frequency Domain - FFT Magnitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

# -----------------------------
# 3) FFT Phase Plot
# -----------------------------
plt.figure()
plt.plot(freqs, phase)
plt.title("Frequency Domain - FFT Phase")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid(True)

plt.show()
