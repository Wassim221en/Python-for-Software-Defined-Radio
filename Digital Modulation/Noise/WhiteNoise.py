import numpy as np
import matplotlib.pyplot as plt

def GenerateWhiteNoise(num):
    N = num
    white_noise = np.random.randn(N)
    return white_noise
def PlotWhiteNosie(white_noise):
    plt.figure(figsize=(10,4))
    plt.plot(white_noise, '.', markersize=3)
    plt.title("White Gaussian Noise (Time Domain)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.figure()
    plt.hist(white_noise, bins=50)
    plt.title("Histogram of White Gaussian Noise")
    plt.show()
