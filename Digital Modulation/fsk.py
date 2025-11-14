import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


data = [1, 0, 1, 0, 1, 0, 1]


f0 = 50    
f1 = 100   
bit_duration = 0.1
fs = 8000   

t = np.array([])  
signal = np.array([])

for bit in data:
    t_bit = np.linspace(0, bit_duration, int(fs * bit_duration), endpoint=False)
    t = np.concatenate((t, t_bit))
    if bit == 0:
        signal = np.concatenate((signal, np.sin(2 * np.pi * f0 * t_bit)))
    else:
        signal = np.concatenate((signal, np.sin(2 * np.pi * f1 * t_bit)))
data_out=[]
samples_per_bit=int(fs*bit_duration)
for i in range(0,len(signal),samples_per_bit):
    bit_singal=signal[i:i+samples_per_bit]
    fft_vals=np.fft.fft(bit_singal)
    fft_freqs=np.fft.fftfreq(len(bit_singal),1/fs)
    peak_index = np.argmax(np.abs(fft_vals[:len(fft_vals)//2]))
    peak_freq = abs(fft_freqs[peak_index])
    if(abs(peak_freq -f0)<abs(peak_freq-f1)):
        data_out.append(0)
    else:
        data_out.append(1)

write("fsk_signal.wav", fs, signal.astype(np.float32))
plt.figure()
plt.step(data_out,'-') 
plt.step(data,'+')
plt.figure()
plt.plot(signal)
plt.figure(figsize=(10,4))
plt.plot(t, signal)
plt.title("BFSK Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

write("fsk_signal.wav", fs, signal.astype(np.float32))
