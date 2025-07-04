'''
https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1243051/full
'''
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)
    
out_path = '/Users/bdd/code/fus_anes/media/chirp_audio'

fs = 44100  # Sampling rate
duration = 0.5  # Duration in seconds
carrier_freq = 1000  # Carrier tone in Hz
mod_start = 55  # Start frequency of modulation (Hz)
mod_end = 25  # End frequency (Hz)
ramp_duration = 0.01  # Onset/offset ramp in seconds

def generate_chirp(kind):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_freq * t)

    mod_freq = np.linspace(mod_start, mod_end, len(t))
    mod_phase = 2 * np.pi * np.cumsum(mod_freq) / fs
    amplitude_envelope = 0.5 * (1 + np.sin(mod_phase))  # Range [0,1]

    chirp_modulated = amplitude_envelope * carrier

    ramp_samples = int(ramp_duration * fs)
    ramp = np.linspace(0, 1, ramp_samples)
    envelope = np.ones_like(chirp_modulated)
    envelope[:ramp_samples] *= ramp
    envelope[-ramp_samples:] *= ramp[::-1]
    chirp_modulated *= envelope

    chirp_modulated /= np.max(np.abs(chirp_modulated))

    # --- white noise ctrl
    white_noise = np.random.normal(0, 1, len(t))
    white_noise = bandpass_filter(white_noise, 800, 1200, fs)  # match chirp band
    white_noise *= envelope
    white_noise /= np.max(np.abs(white_noise))

    if kind == 'chirp':
        return chirp_modulated
    elif kind == 'noise':
        return white_noise

def save_chirp(filename, tone):
    # Normalize to 16-bit PCM
    tone /= np.max(np.abs(tone))
    tone_int16 = np.int16(tone * 32767)
    
    os.makedirs(out_path, exist_ok=True)

    path = os.path.join(out_path, filename)
    wavfile.write(path, 44100, tone_int16)

    print(f"Saved {filename}")

if __name__ == "__main__":
    save_chirp('chirp.wav', generate_chirp('chirp'))
    save_chirp('chirp_white.wav', generate_chirp('noise'))


