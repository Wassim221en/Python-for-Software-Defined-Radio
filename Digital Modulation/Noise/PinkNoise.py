from scipy.signal import lfilter
from WhiteNoise import GenerateWhiteNoise,PlotWhiteNosie
import matplotlib.pyplot as plt

b = [0.05, 0.1, 0.5]
a = [1, -0.9, 0.3]
white_noise=GenerateWhiteNoise(1000)
pink_noise = lfilter(b, a, white_noise)
plt.plot(pink_noise, '.', markersize=3)
plt.title("Approximate Pink Noise (Time Domain)")
plt.show()
