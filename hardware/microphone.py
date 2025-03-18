import numpy as np
import os
import multiprocessing as mp
import queue
import threading
import logging
import h5py
import ctypes
import time
import pyaudio

import fus_anes.config as config
from fus_anes.util import now, now2

def probe_mics():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


def play_audio(filename):
    with h5py.File(filename, 'r') as h:
        data = np.array(h['audio'])

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)
    stream.write(data.astype(np.int16).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


if config.THREADS_ONLY:
    mproc = threading.Thread
else:
    mproc = mp.Process

class Microphone(mproc):
    def __init__(self, name, error_queue=None):
        super(Microphone, self).__init__()
        self.save_path = os.path.join(config.data_path, f'{name}_microphone.h5')
        self.error_queue = error_queue or queue.Queue()

        self.save_audio_buffer_shape = config.audio_save_chunk
        self.hdf_resize_audio = config.audio_hdf_resize
        
        self.kill_flag = mp.Value('b', 0)
        self._on = mp.Value('b', 0)
        self._saving = mp.Value('b', 0)

        self.current_audio_q = mp.Queue()
        self.current_audio = np.zeros(int(config.n_audio_display * 44100), dtype=np.int16)
        self.current_audio_mp = mp.Array(ctypes.c_int16, self.current_audio)

        self.n_frames_captured_a = mp.Value('i', 0)
        self.n_frames_queued_a = mp.Value('i', 0)
        self.n_frames_saved_a = mp.Value('i', 0)
        
        self.audio_buffer = mp.Queue()
        
        threading.Thread(target=self.keep_current_audio, daemon=True).start()
        self.start()

    def setup_audio(self):
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=44100,
                                    input=True,
                                    stream_callback=self.audio_callback,
                                    input_device_index=config.audio_device_idx,
                                    frames_per_buffer=config.audio_stream_chunk)

    def audio_callback(self, data, n, tinfo, flags):
        if self.kill_flag.value:
            return None, pyaudio.paComplete

        ts = now()
        dat = np.frombuffer(data, dtype=np.int16)
        if self._saving.value:
            self.audio_buffer.put([dat, ts])
            self.current_audio_q.put(dat)
            self.n_frames_captured_a.value += len(dat)

        return None, pyaudio.paContinue
        
    def run(self):
        try:
            self.setup_audio()

            empty_a = np.zeros(self.hdf_resize_audio, dtype=np.int16)
            empty_a_t = np.zeros(self.hdf_resize_audio, dtype=np.float64)

            with h5py.File(self.save_path, 'a') as hfile:
                if 'audio' not in hfile:
                    ds_a = hfile.create_dataset('audio', data=empty_a,
                                                compression='lzf', dtype=np.int16,
                                                maxshape=(None,))
                else:
                    ds_a = hfile['audio']
                
                if 'audio_time' not in hfile:
                    ds_a_t = hfile.create_dataset('audio_time', data=empty_a_t,
                                                compression='lzf', dtype=np.float64,
                                                maxshape=(None,))
                else:
                    ds_a_t = hfile['audio_time']

            self.hdf_idx_a = 0
            n_dumped_audio = 0

            self.save_audio_buffer = np.zeros(self.save_audio_buffer_shape, dtype=np.int16)
            self.save_audio_buffer_t = np.zeros(self.save_audio_buffer_shape, dtype=np.float64)

            self._saving.value = 1
            self._on.value = 1
            finished = False

            while True:
                try:
                    aud, ts = self.audio_buffer.get(block=False)
                    if len(aud) < config.audio_stream_chunk:
                        aud = np.pad(aud, (0, config.audio_stream_chunk - len(aud)), 'constant', constant_values=(0, 0)).astype(np.int16)
                    self.save_audio_buffer[n_dumped_audio : n_dumped_audio + config.audio_stream_chunk] = aud
                    self.save_audio_buffer_t[n_dumped_audio : n_dumped_audio + config.audio_stream_chunk] = ts
                    n_dumped_audio += config.audio_stream_chunk
                    self.n_frames_queued_a.value += len(aud)

                    if self.n_frames_captured_a.value - self.n_frames_saved_a.value > self.save_audio_buffer_shape * 3:
                        logging.warning(f'Falling behind on audio saving:\nCaptrd\t{self.n_frames_captured_a.value}\nQueued\t{self.n_frames_queued_a.value}\nSaved\t{self.n_frames_saved_a.value}\nDelta\t{self.n_frames_captured_a.value-self.n_frames_saved_a.value}\n')

                    if n_dumped_audio == config.audio_save_chunk:
                        self.empty_save_buffer()
                        n_dumped_audio = 0
                        
                except queue.Empty:
                    if self.kill_flag.value:
                        if not finished:
                            self.empty_save_buffer(N=n_dumped_audio, final=True)
                            finished = True
                
                if finished:
                    while self.audio_stream.is_active():
                        time.sleep(0.010)
                    self.audio_stream.close()
                    self.pa.terminate()
                    self._on.value = False
                    break

        except Exception as e:
            self.error_queue.put(f'Microphone main: {str(e)}')
            
    def empty_save_buffer(self, N=None, final=False):

        dat, dat_t = self.save_audio_buffer, self.save_audio_buffer_t

        if N is not None:
            dat = dat[:N]
            dat_t = dat_t[:N]

        with h5py.File(self.save_path, 'a') as hfile:
            ds = hfile['audio']
            ds_t = hfile['audio_time']
            hdf_idx = self.hdf_idx_a
            hdfrs = self.hdf_resize_audio
            
            if hdf_idx + len(dat) > ds.shape[0]:
                ds.resize(ds.shape[0] + hdfrs, axis=0)
                ds_t.resize(ds.shape[0] + hdfrs, axis=0)
            ds[hdf_idx:hdf_idx+len(dat)] = dat
            ds_t[hdf_idx:hdf_idx+len(dat)] = dat_t
            
            self.hdf_idx_a += len(dat)
            self.n_frames_saved_a.value = hdf_idx
        
            # cut off extra
            if final:
                hdf_idx = self.hdf_idx_a
                ds.resize(hdf_idx, axis=0)
                ds_t.resize(hdf_idx, axis=0)
    
    def keep_current_audio(self):  
        while not self.kill_flag.value:    
            try:
                aud = self.current_audio_q.get(block=True)                         
                self.current_audio = np.roll(self.current_audio, -len(aud))
                self.current_audio[-len(aud):] = aud
                self.current_audio_mp[:] = self.current_audio.copy()
            except queue.Empty:
                pass
               
    def get_current_audio(self):            
        return np.frombuffer(self.current_audio_mp.get_obj(), dtype=np.int16)

    def end(self):
        self.kill_flag.value = 1
        while self._on.value:
            time.sleep(0.010)
