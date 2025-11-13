import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

N = 1000 
symbols = np.array([-1+1j, -1-1j, 1+1j, 1-1j]) 
data = np.random.choice(symbols, N)
SNR_dB = 5 
signal_power = np.var(data)
noise_power = signal_power / (10**(SNR_dB/10))
white_noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j*np.random.randn(N))
white_real = np.random.randn(N)
white_imag = np.random.randn(N)

b = [0.05, 0.1, 0.5]
a = [1, -0.9, 0.3]

pink_real = lfilter(b, a, white_real)
pink_imag = lfilter(b, a, white_imag)

pink_noise = (pink_real + 1j*pink_imag)
pink_noise = pink_noise * np.sqrt(noise_power / np.var(pink_noise))
complex_noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j*np.random.randn(N))


data_white = data + white_noise
data_pink = data + pink_noise
data_complex = data + complex_noise
plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.plot(np.real(data_white), np.imag(data_white), '.')
plt.title("QPSK + White Noise")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)

plt.subplot(1,4,2)
plt.plot(np.real(data_pink), np.imag(data_pink), '.')
plt.title("QPSK + Pink Noise")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)

plt.subplot(1,4,3)
plt.plot(np.real(data_complex), np.imag(data_complex), '.')
plt.title("QPSK + Complex Noise")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)

plt.subplot(1,4,4)
plt.plot(np.real(data), np.imag(data), '.')
plt.title("QPSK")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.show()
