import matplotlib.pyplot as plt
import numpy as np
N=1000
complex_noise = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)

plt.plot(np.real(complex_noise), '.', label='Real')
plt.plot(np.imag(complex_noise), '.', label='Imag')
plt.legend()
plt.title("Complex Gaussian Noise")
plt.show()


plt.figure(figsize=(5,5))
plt.plot(np.real(complex_noise), np.imag(complex_noise), '.')
plt.title("Complex Noise on IQ Plot")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.show()
