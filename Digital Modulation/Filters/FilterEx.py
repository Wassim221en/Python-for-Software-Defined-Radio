import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz, correlate
from scipy.fftpack import fft, fftshift

# ---------------- Parameters (اختَرْت قيم افتراضية — سأشرح الأساس لكل واحدة أسفل الكود) ----------------
fs = 1_000_000.0      # sample rate [Hz]
fc = 100_000.0        # carrier frequency [Hz]
bit_rate = 5_000.0    # bits per second
num_bits = 160        # عدد البتات
snr_db = 25.0         # SNR (dB) بالنسبة لإشارة الـ ASK
interf_freq = 300_000.0  # تداخل عند 300 kHz
interf_amp = 1.0      # سعة المزعج (بالنسبة لإشارة الحامل)
numtaps_bp = 401      # عدد تابات فلتر band-pass (اختيار فردي)
bp_window = 'blackman' # نافذة الـ FIR للـ band-pass
lp_taps = 201         # تابات فلتر low-pass (للإشارة الأساسية بعد الضرب)
lp_window = 'hann'

# ---------------- Derived values ----------------
Ts = 1.0 / fs
samples_per_bit = int(fs / bit_rate)
t_total = num_bits / bit_rate
N = int(np.floor(t_total * fs))
t = np.arange(N) * Ts

# ---------------- Generate random bits and NRZ baseband ----------------
np.random.seed(1)
bits = np.random.randint(0, 2, size=num_bits)
# NRZ pulses (0->0, 1->1). For better robustness we could use bipolar (-1, +1) but keep OOK/ASK as requested.
baseband = np.repeat(bits, samples_per_bit)[:N].astype(float)

# ---------------- Create ASK (OOK) signal ----------------
carrier = np.cos(2 * np.pi * fc * t)
ask_signal = baseband * carrier  # OOK: carrier present when bit==1

# ---------------- Add high-frequency interferer and AWGN ----------------
interferer = interf_amp * np.cos(2 * np.pi * interf_freq * t)
# scale noise to get requested SNR relative to ASK signal power
signal_power = np.mean(ask_signal**2) + 1e-16
snr_linear = 10**(snr_db / 10.0)
noise_power = signal_power / snr_linear
noise = np.sqrt(noise_power) * np.random.randn(N)

mixed = ask_signal + interferer + noise

# ---------------- Design band-pass FIR filter around fc to isolate carrier ----------------
# Choose passband wide enough to include modulation sidebands: take +/- (2 * bit_rate) or more
sideband_margin = 8 * bit_rate  # margin to include main sidebands
f1 = max(0.0, fc - sideband_margin)
f2 = min(fs/2.0 * 0.9999, fc + sideband_margin)
h_bp = firwin(numtaps_bp, [f1, f2], pass_zero=False, fs=fs, window=bp_window)

# apply band-pass
filtered_bp = lfilter(h_bp, 1.0, mixed)

# correct group delay of band-pass (FIR linear phase)
grp_delay_bp = (numtaps_bp - 1) // 2
# shift signal to compensate (pad beginning with zeros)
filtered_bp_corrected = np.roll(filtered_bp, -grp_delay_bp)
# zero pad end (because rolled content wraps around)
filtered_bp_corrected[-grp_delay_bp:] = 0.0

# ---------------- Coherent demodulation (multiply by local carrier and low-pass) ----------------
local_cos = np.cos(2 * np.pi * fc * t)
local_sin = np.sin(2 * np.pi * fc * t)  # for quadrature if needed

# Multiply (mix) to baseband
i_mixed = filtered_bp_corrected * local_cos
q_mixed = filtered_bp_corrected * local_sin  # q should be ~0 if phase aligned

# Design low-pass to keep baseband (cutoff somewhat > bit_rate to pass shape)
lp_cutoff = 1.2 * bit_rate  # Hz
h_lp = firwin(lp_taps, lp_cutoff, fs=fs, window=lp_window)
i_base = lfilter(h_lp, 1.0, i_mixed)
q_base = lfilter(h_lp, 1.0, q_mixed)
grp_delay_lp = (lp_taps - 1) // 2
i_base = np.roll(i_base, -grp_delay_lp); i_base[-grp_delay_lp:] = 0.0
q_base = np.roll(q_base, -grp_delay_lp); q_base[-grp_delay_lp:] = 0.0

# magnitude (coherent envelope) - if phase aligned Q ~ 0 and I holds amplitude
envelope_coherent = np.sqrt(i_base**2 + q_base**2)
plt.figure()
plt.plot(envelope_coherent)
# ---------------- Matched filter (integrate-and-dump) across each bit period ----------------
matched = np.ones(samples_per_bit)  # matched filter for NRZ rectangular pulse
plt.figure("matched")
plt.plot(matched)
mf_out = np.convolve(envelope_coherent, matched, mode='same')
plt.figure("mf_out")
plt.plot(mf_out)
# compensate matched filter delay: center of window
mf_delay = (len(matched) - 1) // 2

# ---------------- Sample and decide (adaptive threshold) ----------------
# Determine sampling points: sample at center of each bit window, but compensate total delay (bp + lp + mf)
total_delay = grp_delay_bp + grp_delay_lp + mf_delay
sample_indices = (np.arange(num_bits) * samples_per_bit + samples_per_bit//2 - total_delay).astype(int)
# remove out-of-bound indices
valid = (sample_indices >= 0) & (sample_indices < len(mf_out))
sample_indices = sample_indices[valid]
bits_expected = bits[:len(sample_indices)]

samples = mf_out[sample_indices]

# adaptive threshold: Otsu-like or mean of two peaks; here use k * (max + min)/2 with noise estimate
thr = 0.5 * (np.max(samples) + np.min(samples))
# better: compute bimodal threshold using K-means (2 clusters)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(samples.reshape(-1,1))
centers = sorted(kmeans.cluster_centers_.flatten())
thr_kmeans = 0.5 * (centers[0] + centers[1])

# choose threshold
threshold = thr_kmeans

recovered = (samples > threshold).astype(int)

# BER
ber = np.mean(recovered != bits_expected)

# ---------------- Diagnostic plots ----------------
def compute_fft(x, fs):
    X = fft(x)
    N = len(X)
    freq = np.linspace(-fs/2, fs/2, N)
    Xs = fftshift(np.abs(X)) / N
    return freq, Xs

plt.figure(figsize=(9,3))
plt.plot(t[:2000], mixed[:2000], label='mixed (time segment)')
plt.title('Mixed signal (time) - short segment')
plt.xlabel('Time [s]')
plt.grid(True)

plt.figure(figsize=(9,3))
f_mixed, X_mixed = compute_fft(mixed, fs)
plt.plot(f_mixed, 20*np.log10(X_mixed + 1e-16))
plt.xlim(-fs/2, fs/2)
plt.ylim([-140, 5])
plt.title('Spectrum of mixed signal (dB)')
plt.xlabel('Frequency [Hz]')

plt.figure(figsize=(9,3))
w, H = freqz(h_bp, worN=4096, fs=fs)
plt.plot(w, 20*np.log10(np.abs(H) + 1e-16))
plt.title('Band-pass filter response (dB)')
plt.xlabel('Frequency [Hz]')
plt.grid(True)

plt.figure(figsize=(9,3))
plt.plot(t[:2000], filtered_bp_corrected[:2000], label='BP filtered (corrected)')
plt.title('Band-pass filtered (time) - short seg')
plt.grid(True)

plt.figure(figsize=(9,3))
plt.plot(t[:2000], envelope_coherent[:2000], label='coherent envelope (I/Q)')
plt.plot(t[:2000], mf_out[:2000]/samples_per_bit, label='matched filter (normalized)')
plt.title('Envelope and matched filter output (short seg)')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10,2))
plt.step(np.arange(len(bits_expected)), bits_expected, where='mid', label='original bits')
plt.step(np.arange(len(recovered)), recovered, where='mid', label='recovered bits')
plt.title(f'Original vs Recovered bits (BER = {ber:.4f})')
plt.legend()
plt.grid(True)

plt.show()

# ---------------- Summary print ----------------
print("Parameters summary:")
print(f"fs = {fs/1e3:.1f} kHz, fc = {fc/1e3:.1f} kHz, bit_rate = {bit_rate/1e3:.1f} kHz")
print(f"interf_freq = {interf_freq/1e3:.1f} kHz, interf_amp = {interf_amp}")
print(f"numtaps_bp = {numtaps_bp}, bp passband = [{f1/1e3:.1f}, {f2/1e3:.1f}] kHz")
print(f"lp_cutoff = {lp_cutoff} Hz, lp_taps = {lp_taps}")
print(f"samples_per_bit = {samples_per_bit}, total bits used = {len(bits_expected)}")
print(f"threshold (kmeans) = {threshold:.4g}, BER = {ber:.4%}")
