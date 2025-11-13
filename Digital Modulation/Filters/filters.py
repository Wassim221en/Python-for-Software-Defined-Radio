import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def LowPassFilter(fs,cutoff,num_taps):
    lpf = signal.firwin(num_taps, cutoff, fs=fs)
    w, h = signal.freqz(lpf, worN=8000)
    plt.figure()
    plt.plot(h, '.-')
    plt.figure()
    plt.plot((fs * 0.5 / np.pi) * w, 20 * np.log10(np.abs(h)))
    plt.title('Low-Pass Filter Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    LowPassFilter(fs=1000,cutoff=100,num_taps=51)
