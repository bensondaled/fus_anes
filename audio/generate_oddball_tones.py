import numpy as np
import os
from scipy.io import wavfile
    
out_path = '/Users/bdd/code/fus_anes/media/oddball_audio'

def generate_tone(freq, duration_ms=100, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    # 5 ms linear ramp fade-in and fade-out
    ramp_len = int(sample_rate * 0.005)
    ramp = np.linspace(0, 1, ramp_len)
    tone[:ramp_len] *= ramp
    tone[-ramp_len:] *= ramp[::-1]
    return tone

def save_tone(filename, freq, duration_ms=150, sample_rate=44100):
    tone = generate_tone(freq, duration_ms, sample_rate)
    # Normalize to 16-bit PCM
    tone /= np.max(np.abs(tone))
    tone_int16 = np.int16(tone * 32767)
    
    os.makedirs(out_path, exist_ok=True)
    path = os.path.join(out_path, filename)

    wavfile.write(path, sample_rate, tone_int16)
    print(f"Saved {filename}")

if __name__ == "__main__":
    sample_rate = 44100
    save_tone('standard_tone.wav', freq=1000, duration_ms=100, sample_rate=sample_rate)
    save_tone('deviant_tone.wav', freq=1200, duration_ms=100, sample_rate=sample_rate)


