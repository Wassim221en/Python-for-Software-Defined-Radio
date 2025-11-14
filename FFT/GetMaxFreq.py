import numpy as np
fs =1e3
t=np.arange(0,1,1/fs)
signal=np.sin(2*np.pi*200*t)+0.5*np.sin(2*np.pi*400*t)
fft_vals=np.fft.fft(signal)
freqs=np.fft.fftfreq(len(signal),1/fs)
dominant_freq = freqs[np.argmax(np.abs(fft_vals))]
print("Dominant Frequency =", dominant_freq)