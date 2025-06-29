import numpy as np
import multiprocessing as mp
import threading
import queue
import time
import warnings
import os
import sys
import ctypes
from pylsl import StreamInlet, resolve_byprop

from fus_anes.util import LiveFilter, now, multitaper_spectrogram, save
MTS = multitaper_spectrogram
import fus_anes.config as config

if config.THREADS_ONLY:
    tcls = threading.Thread
else:
    tcls = mp.Process

def downsample_spect(spect, freqs):
    factor = config.spect_freq_downsample
    assert len(freqs) % factor == 0, f'Make spect freq downsample a factor of f{len(freqs)}'
    shaped = spect.reshape(spect.shape[0], spect.shape[1] // factor,
                                    factor, spect.shape[2])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        spect = np.nanmean(shaped, axis=2)
    freqs = np.mean(freqs.reshape(-1, factor), axis=1)
    return spect, freqs

class EEG(tcls):
    # NOTE that spect_memory is in dB and is normalized to turn it to int8!
    
    def __init__(self,
                 # EEG hardware
                 n_channels=config.n_channels,
                 fs=config.fs,
                 read_buffer_length=config.read_buffer_length, # samples
                 raw_dtype=np.float64,
                 spect_dtype=np.float64, # for mp arrays, float32 uses "f", float64 uses "d"
                
                 # Raw EEG memory
                 n_live_chan=config.n_live_chan,
                 memory_length=config.eeg_memory_length, # samples

                 # Spectrograms
                 spect_interval=config.spect_update_interval, # samples
                 spect_memory_length=config.spect_memory_length, # secs

                 # Saving
                 save_buffer_length=config.save_buffer_length,
                 saver_obj_buffer=None,
                 error_queue=None,

                 ):
        super(EEG, self).__init__()

        assert spect_interval <= memory_length
        assert spect_interval % read_buffer_length == 0
                
        self.n_timefields = len(now()) + 2 # the 2 is for the inlet pull_sample time and the time correction
        assert self.n_timefields - 2 == 3 # see assertion below where this matters
        
        # EEG hardware
        self.n_channels = n_channels
        self.fs = fs
        self.read_buffer_length = read_buffer_length
        self.raw_dtype = raw_dtype
        self.spect_dtype = spect_dtype
        
        # Raw trace memory storage
        self.n_live_chan = n_live_chan
        self.memory_dims = [memory_length,
                            n_channels+self.n_timefields] # +4 for time
        self.memory = np.zeros(self.memory_dims).astype(self.raw_dtype)
        self.memory_mp = mp.Array('d', self.memory.copy().ravel())
        self.memory_idx = mp.Value('i', 0)

        # Spectrogram live computation
        self.spect_interval = spect_interval
        winstep = spect_interval / self.fs
        self.spect_params = dict(window_size=winstep,
                                 window_step=winstep,
                                 )
        # Spectrogram memory storage
        dummy_input = self.memory[-self.spect_interval:, :-self.n_timefields].copy()
        dummy, dummyt, dummyf = self.compute_dummy_spect(dummy_input)
        self.spect_memory_tf = [dummyt, dummyf]
        spect_memory_length = int(round(spect_memory_length / winstep)) # secs to windows
        self.spect_memory_dims = [self.n_channels, # or n_live_chan
                                  dummy.shape[0],
                                  spect_memory_length] # N x freq x wins
        self.spect_memory = (np.zeros(self.spect_memory_dims) * np.nan).astype(self.spect_dtype)
        self.spect_memory_mp = mp.Array('d', self.spect_memory.copy().ravel())

        # Filtering
        self.filter_params = mp.Array('d', [0,0,0])
        self.new_filter_params_flag = mp.Value('b', 0)
        
        # Saving
        self.eeg_queue = mp.Queue()
        self.save_buffer_length = save_buffer_length
        self.save_buffer_dims = [save_buffer_length, n_channels + self.n_timefields]
        self.n_new = 0 # n added to save buffer
        self.n_not_proc = 0 # n added since last processing
        self.save_buffer = np.zeros(self.save_buffer_dims).astype(self.raw_dtype)
        self.saver_obj_buffer = saver_obj_buffer
        self.error_queue = error_queue
        
        # Runtime
        self._on = mp.Value('b', True)
        self.kill_flag = mp.Value('b', False)
        self.start_proc_flag = mp.Value('b', False)
        self.first_spect_time = mp.Value('d', -1)
        self.processing_queue = mp.Queue()
        self.i = 0 # temp
        self.total_hardware_writes = mp.Value('i', 0)
        self.total_dequeues = 0

        # Start
        tcls(target=self.stream_eeg, daemon=True).start() 
        tcls(target=self.live_spect_processing, daemon=True).start()
        time.sleep(0.250)
        self.start()

    def start_processing(self):
        self.start_proc_flag.value = True

    def compute_dummy_spect(self, dummy_input):
        dummy_input[:] = np.random.random(size=dummy_input.shape)
        dummy, dummyt, dummyf = MTS(dummy_input,
                                    fs=self.fs,
                                    **self.spect_params)
        fi_min, fi_max = 0, np.argmin(np.abs(dummyf-config.max_freq))
        self.spect_f_slice = slice(fi_min, fi_max)
        dummyf = dummyf[self.spect_f_slice]
        dummy = dummy[:, self.spect_f_slice, :]
        dummy, dummyf = downsample_spect(dummy, dummyf)
        dummy = np.nanmean(dummy, axis=0)
        return dummy, dummyt, dummyf

    def stream_eeg(self):
        '''
        in reality this will be whatever the api does to read new data, and will dump into the queue as below (ideally it'll be a callback that gets autocalled when new data is available)

        https://github.com/labstreaminglayer/pylsl/blob/master/pylsl/examples/ReceiveDataInChunks.py
        '''
        try:
            if not config.SIM_DATA:
                streams = resolve_byprop('type', 'EEG')
                inlet = StreamInlet(streams[0])
                read_buffer = np.zeros([config.read_buffer_length, config.n_channels+self.n_timefields])
                rbi = 0
                while self._on.value and not self.kill_flag.value:
                    dat, tse = inlet.pull_sample()
                    tc = inlet.time_correction()
                    
                    ts = now()
                    read_buffer[rbi, :] = dat + [tse, tc] + ts
                    rbi += 1
                    if rbi == config.read_buffer_length:
                        self.eeg_queue.put(read_buffer.copy())
                        rbi = 0
                        self.total_hardware_writes.value += 1
                inlet.close_stream()
            
            elif config.SIM_DATA:
                ex_data = np.load(os.path.join(config.data_path, 'ex_eeg.npy'))
                ex_data *= 10
                ex_data += 0.01 * np.sin(2*np.pi*60*np.arange(0, len(ex_data))/config.fs) # add 60hz noise
                ex_data += np.random.normal(0, 10.0, size=ex_data.shape) # add gaussian noise
                idx_arr = np.random.randint(len(ex_data)-config.read_buffer_length, size=self.n_channels)
                while self._on.value and not self.kill_flag.value:
                    t0 = now(minimal=True)
                    
                    '''
                    t = np.arange(self.i, self.i+self.read_buffer_length) / self.fs
                    self.i += self.read_buffer_length
                    fbase = self.i/20000 # to make drift in freq over time
                    dat = [np.sin(2*np.pi*i*1.*fbase*t) + 2*np.cos(2*np.pi*i*2.*fbase*t) for i in range(1, self.n_channels+1)]
                    dat = np.array(dat).astype(self.raw_dtype)
                    dat += np.random.normal(0, 0.8, size=dat.shape)
                    '''

                    dat = []

                    for _i,i in enumerate(idx_arr):
                        dat_ = ex_data[i:i+config.read_buffer_length]
                        dat.append(dat_)
                    ex_data = np.roll(ex_data, -config.read_buffer_length)
                    dat = np.array(dat).astype(self.raw_dtype)

                    ts = now()
                    dat = np.concatenate([dat, [[0]*dat.shape[1], [0]*dat.shape[1], [ts[0]]*dat.shape[1], [ts[1]]*dat.shape[1], [ts[2]]*dat.shape[1] ]])
                    dat = dat.T

                    self.eeg_queue.put(dat)
                    self.total_hardware_writes.value += 1
                    
                    time.sleep(max(0,self.read_buffer_length / self.fs - (now(minimal=True)-t0)))
        except Exception as e:
            self.error_queue.put(f'EEG streaming: {str(e)}')

    def run(self):
        sent_behind_error = False
        try:
            lfilt = LiveFilter()
            while self._on.value:
                
                # check if any new data dumped into queue from hardware
                try:
                    dat = self.eeg_queue.get(block=False)
                    self.total_dequeues += 1
                    if (self.total_hardware_writes.value - self.total_dequeues) > 5 and self.total_dequeues>1000:
                        print(f'Read {self.total_hardware_writes.value}, wrote {self.total_dequeues}')
                        if not sent_behind_error:
                            self.error_queue.put(f'EEG main: ' + f'Read {self.total_hardware_writes.value}, wrote {self.total_dequeues}') 
                            sent_behind_error = True
                except queue.Empty:
                    if self.kill_flag.value:
                        self.empty_save_buffer(self.n_new)
                        self._on.value = False
                    continue

                # dump new data into saving buffer
                self.save_buffer = np.roll(self.save_buffer,
                                           -self.read_buffer_length,
                                           axis=0)
                self.save_buffer[-self.read_buffer_length:, :] = dat[:, :].copy()
                self.n_new += self.read_buffer_length
                self.n_not_proc += self.read_buffer_length

                #-- process new data (specifically done after saving buffer, so completely raw data are saved)

                # reference
                dat[:,:-self.n_timefields] = dat[:,:-self.n_timefields] - dat[:, config.chan_reference][:, None]

                # filter
                if self.new_filter_params_flag.value:
                    lo, hi, notch = self.filter_params.get_obj()
                    self.new_filter_params_flag.value = 0
                    lfilt.lo = lo
                    lfilt.hi = hi
                    lfilt.notch = notch
                    lfilt.refresh_filter()
                dat[:,:-self.n_timefields] = lfilt(dat[:,:-self.n_timefields])
                
                # update memory with new data
                self.memory = np.roll(self.memory,
                                      -self.read_buffer_length,
                                      axis=0)
                slice_to_set = (slice(-self.read_buffer_length, None),
                                slice(None, None))
                self.memory[slice_to_set] = dat[:, :].copy()
                self.memory_mp[:] = self.memory.copy().ravel()
                self.memory_idx.value += self.read_buffer_length

                # compute next spect if enough new data are present
                if (self.start_proc_flag.value==True) and (self.n_not_proc >= self.spect_interval):
                    doi = self.memory[-self.spect_interval:].copy()
                    self.processing_queue.put(doi)
                    self.n_not_proc = 0
                
                # empty the buffer to file if it's full
                if self.n_new >= self.save_buffer_length:
                    if self.n_new > self.save_buffer_length:
                        warnings.warn(f'Lost some EEG samples: {n_new} were acquired, {self.save_buffer_length} saved') # shouldnt happen
                    self.empty_save_buffer()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.error_queue.put(f'EEG main (line {exc_tb.tb_lineno}): {str(e)}')

    def slice_to_flat_idxs(self, sls, shape):
        rngs = []
        for didx, sl in enumerate(sls):
            sl_0 = sl.start or 0
            if sl_0 < 0:
                sl_0 += shape[didx]
            sl_1 = sl.stop or shape[didx]
            sl_r = np.arange(sl_0, sl_1)
            rngs.append(sl_r)

        gs = np.meshgrid(*rngs)
        rav = tuple([g.ravel() for g in gs])
        stack = np.column_stack(rav).T

        mi = np.ravel_multi_index(stack, shape)
        mi = mi.tolist()
        return mi

    def empty_save_buffer(self, N=0):
        if self.saver_obj_buffer is None:
            return
        
        save('eeg',
             data=self.save_buffer[-N:, :-3].copy(), # the two added columns of hardware ts (inlet and offset), followed by 3 general now() fields. we cut off the 3 and use them below, but we keep the 2 former
             buffer=self.saver_obj_buffer,
             time_data=[ # note that if n_timefields changes, this needs manual adjustment - it assumes 3 fields come from now(), assertion flag
                            self.save_buffer[-N:, -3].copy(),
                            self.save_buffer[-N:, -2].copy(),
                            self.save_buffer[-N:, -1].copy(),
                 ],
             columns=[f'{i}' for i in range(self.n_channels)]+['hardware_ts', 'hardware_offset'],
           )
        self.n_new = 0
    
    def get_memory(self, with_idx=False):
        if with_idx == False:
            return np.frombuffer(self.memory_mp.get_obj(), dtype=self.raw_dtype).reshape(self.memory_dims).astype(self.raw_dtype)
        elif with_idx == True:
            # keeps track of how many added since last time with_idx was requested
            idx = self.memory_idx.value
            self.memory_idx.value = 0
            return np.frombuffer(self.memory_mp.get_obj(), dtype=self.raw_dtype).reshape(self.memory_dims).astype(self.raw_dtype), idx

    def get_spect_memory(self):
        # NOTE that this is in dB (logged) for performance
        return (np.frombuffer(self.spect_memory_mp.get_obj(), dtype=self.spect_dtype).reshape(self.spect_memory_dims)).astype(self.spect_dtype)

    def live_spect_processing(self):
        spect_memory_idx = 0
        while self._on.value:

            try:
                dat = self.processing_queue.get(block=False)
            except queue.Empty:
                continue
            
            if self.first_spect_time.value == -1:
                self.first_spect_time.value = dat[0, -2] # uses wall clock now() taken at eeg's acquisition as the reference time

            dat = dat[:, :-self.n_timefields] # remove timestamp columns

            spect, s_time, s_freq = MTS(dat, fs=self.fs,
                                        **self.spect_params)
            
            spect = spect[:, self.spect_f_slice, :] # N x freq x wins
            spect, _ = downsample_spect(spect, self.spect_memory_tf[1])

            # NOTE conversion to power here, because performance is better when done here and not in interface, because here it only needs to be done once ever, vs repeatedly over the full spect_memory
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'divide by zero encountered in log10')
                if config.spect_log:
                    spect = 10 * np.log10(spect) 
            
            # place new data at next spot from beginning
            # TODO: this will throw error when max spect memory size is reached - need a mechanism to start rolling / extending / throwing out data
            i = spect_memory_idx
            slice_to_set = (slice(None, None), # N live channels
                            slice(None, None), # freqs
                            slice(i, i+spect.shape[-1]) # windows
                            )
            self.spect_memory[slice_to_set] = spect[:,:,:]
            mi = self.slice_to_flat_idxs(slice_to_set,
                                         self.spect_memory.shape)
            newvals = self.spect_memory.ravel()[mi].copy().tolist()
            for idx, nv in zip(mi, newvals):
                self.spect_memory_mp[idx] = nv
            spect_memory_idx += spect.shape[-1]

    def set_filters(self, lo=config.eeg_lopass, hi=config.eeg_hipass, notch=config.eeg_notch):
        self.filter_params[:] = [lo, hi, notch]
        self.new_filter_params_flag.value = 1

    def end(self):
        self.kill_flag.value = True
        while self._on.value:
            pass

