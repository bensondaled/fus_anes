import numpy as np
import sounddevice as sd
import threading

import fus_anes.config as config
from fus_anes.util import now

if config.audio_backend == 'ptb':
    from psychopy import sound, prefs
    prefs.hardware['audioLatencyMode'] = 3
    prefs.hardware['audioLib'] = ['PTB']

    from psychtoolbox import audio
    import psychtoolbox as ptb
    ptb.PsychPortAudio('Initialize')
    pahandle = ptb.PsychPortAudio('Open')

def end_audio():
    if self.audio_backend == 'ptb':
        try:
            ptb.PsychPortAudio('Close', pahandle)
        except:
            pass

def probe_audio_devices():
    print(sd.query_devices())

def play_tone_precisely(tone_data, fs):
    if config.audio_backend == 'sounddevice':
        return play_tone_precisely_sd(tone_data, fs)
    elif config.audio_backend == 'ptb':
        return play_tone_precisely_ptb(tone_data, fs)

def play_tone_precisely_ptb(tone_data, fs, play_after=0.100):
    ''' doesnt work right despite being on psychopy website as primary suggestion
    now_ptb = ptb.GetSecs()
    tone_data.play(when=now_ptb + play_after)
    return now_internal + play_after
    '''
    waveform = tone_data._getSamples()
    ptb_waveform = waveform.T
    tone_data = np.array([ptb_waveform, ptb_waveform])

    ptb.PsychPortAudio('FillBuffer', pahandle, audiodata)
    ptb.PsychPortAudio('Start', pahandle)

    now_internal = now(minimal=True)
    now_ptb = ptb.GetSecs()
    start_time = now_ptb + play_after
    ptb.PsychPortAudio('Start', pahandle, 1, start_time, 0)

    while True:
        status = ptb.PsychPortAudio('GetStatus', pahandle)
        if status['Active']:
            actual_start = status['StartTime']
            print(f"Actually started at: {actual_start:.6f} (PTB time)")
            break

    # === Wait for playback to complete ===
    while True:
        status = ptb.PsychPortAudio('GetStatus', pahandle)
        if not status['Active']:
            break

    ptb.PsychPortAudio('Stop', pahandle, 1)


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

