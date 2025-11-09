import numpy as np
import matplotlib.pyplot as plt


def convert_to_bits(file_path):
    with open(file_path, 'rb') as f: 
        file_bytes = f.read()
    bits = np.unpackbits(np.frombuffer(file_bytes, dtype=np.uint8))
    return bits

def generate_carrier(fc, Tb, fs):
    t_bit = np.arange(0, Tb, 1/fs)
    carrier = np.cos(2 * np.pi * fc * t_bit)
    return carrier, t_bit


def ask_modulate(bits, carrier, A1=1.0, A0=0.2):
    signal = np.array([])
    for b in bits:
        if b == 1:
            signal = np.concatenate((signal, A1 * carrier))
        else:
            signal = np.concatenate((signal, A0 * carrier))
    return signal

def ask_demodulate(signal, Tb, fs, threshold=0.5):
    N = int(Tb * fs) 
    num_bits = len(signal) // N
    bits = []

    for i in range(num_bits):
        bit_chunk = signal[i*N : (i+1)*N]
        avg_amp = np.mean(np.abs(bit_chunk))
        if avg_amp > threshold:
            bits.append(1)
        else:
            bits.append(0)
    return np.array(bits)

def plot_ask(bits, signal, t_bit, fs, Tb):
    T = np.arange(0, len(signal)) / fs
    plt.figure(figsize=(12, 6))

    # Digital bits
    plt.subplot(2, 1, 1)
    bit_signal = np.repeat(bits, len(t_bit))
    plt.plot(T[:len(bit_signal)]*1e3, bit_signal, color='black')
    plt.title("Digital Bits")
    plt.ylabel("Value (0 or 1)")
    plt.xlabel("Time (ms)")
    plt.xlim(0, len(bits)*Tb*1e3)
    for i in range(len(bits)):
        plt.axvline(i*Tb*1e3, color='gray', linestyle='--', alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(T*1e3, signal, color='blue')
    plt.title("ASK Modulated Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.xlim(0, len(bits)*Tb*1e3)
    for i in range(len(bits)):
        plt.axvline(i*Tb*1e3, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()



if __name__ == "__main__":
    file_path = "/home/wassim221e/Desktop/x.json"
    bits = convert_to_bits(file_path)[:20]  

    fs = 1e6
    fc = 100e3
    Tb = 1e-4

    carrier, t_bit = generate_carrier(fc, Tb, fs)
    signal = ask_modulate(bits, carrier, A1=1.0, A0=0.2)
    plot_ask(bits, signal, t_bit, fs, Tb)
    new_bits=recovered_bits = ask_demodulate(signal, Tb=Tb, fs=fs, threshold=0.5)
    errors = 0
    for i in range(len(bits)):
        if bits[i] != new_bits[i]:
            errors += 1
            print(f"Bit {i} mismatch: original={bits[i]} recovered={new_bits[i]}")

    if errors == 0:
        print("ASK demodulation successful! All bits match.")
    else:
        print(f"ASK demodulation failed for {errors} bits.")
    plot_ask(new_bits, signal, t_bit, fs, Tb)
    plt.show()

