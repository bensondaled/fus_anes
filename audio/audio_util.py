import numpy as np
import sounddevice as sd
import fus_anes.config as config

if config.audio_backend == 'ptb':
    import psychtoolbox as ptb
    from psychopy import sound

from fus_anes.util import now

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

    buffer = tone_data.copy()
    onset_time = []

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
            raise sd.CallbackStop()

    with sd.OutputStream(samplerate=fs, channels=1, callback=callback, dtype='float32'):
        sd.sleep(int(len(tone_data) / fs * 1000) + 20)

    return onset_time[0] if onset_time else None

