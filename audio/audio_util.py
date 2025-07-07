import numpy as np
import sounddevice as sd
import threading
import fus_anes.config as config

if config.audio_backend == 'ptb':
    import psychtoolbox as ptb
    from psychopy import sound
    prefs.hardware['audioLatencyMode'] = 3
    prefs.hardware['audioLib'] = ['PTB']

from fus_anes.util import now

def probe_audio_devices():
    print(sd.query_devices())

def play_tone_precisely(tone_data, fs):
    if config.audio_backend == 'sounddevice':
        return play_tone_precisely_sd(tone_data, fs)
    elif config.audio_backend == 'ptb':
        return play_tone_precisely_ptb(tone_data, fs)

def play_tone_precisely_ptb(tone_data, fs, play_after=0.150):
    now_internal = now(minimal=True)
    now_ptb = ptb.GetSecs()
    tone_data.play(when=now_ptb + play_after)

    return now_internal + play_after

def play_tone_precisely_sd(tone_data, fs):
    tone_data = np.asarray(tone_data, dtype=np.float32)
    if tone_data.ndim == 1:
        tone_data = tone_data[:, np.newaxis]
    
    # needed to help prevent clicks on some external speakers and headphones:
    silence = np.zeros((int(0.020 * fs), tone_data.shape[1]))
    tone_data = np.concatenate([tone_data, silence], axis=0)

    buffer = tone_data.copy()
    onset_time = []
    done = threading.Event()

    def callback(outdata, frames, time_info, status):
        nonlocal buffer
        if status:
            print("Playback status:", status)

        if not onset_time:
            onset_time.append(now(minimal=True))

        outdata.fill(0)
        n = min(len(buffer), frames)
        outdata[:n] = buffer[:n]
        buffer = buffer[n:]

        if len(buffer) == 0:
            done.set()

    with sd.OutputStream(samplerate=fs, channels=1, callback=callback, dtype='float32', device=config.audio_in_ch_out_ch[1]):
        done.wait()
        #sd.sleep(int(len(tone_data) / fs * 1000) + 20)

    return onset_time[0] if onset_time else None

