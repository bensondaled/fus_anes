import sys
import time
import multiprocessing as mp
import threading
import queue

import h5py
import json
import tables
import numpy as np
import pandas as pd

from fus_anes.util import now, now2
import fus_anes.config as config

def parse_config_dict(d):
    exclude = ['__name__',
               '__doc__',
               '__package__',
               '__loader__',
               '__spec__',
               '__file__',
               '__cached__',
               '__builtins__',
               ]
    return {k:v for k,v in d.items() if k not in exclude}

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__()
        return json.JSONEncoder.default(self, obj)

class Saver(mp.Process):
#class Saver(threading.Thread): # for testing
    # To save, call saver.write() with either a dict or a numpy array
    # To end, use the saver.end method. It will raise a kill flag, then perform a final flush.

    def __init__(self,
                 session_id,
                 session_obj=None,
                 data_file='',
                 buffer_size=5, # if making large, consider special condition for EEG/other large data that gets written in big chunks, to dump more frequently
                 error_queue=None,
                 ):
        super(Saver, self).__init__()

        self.session_id = session_id
        self.session_obj = session_obj
        self.data_file = data_file
        self.buffer_size = buffer_size
        
        self.error_queue = error_queue

        self.buffer = mp.Queue() # grab bag for anything asked to be saved
        self.kill_flag = mp.Value('b', False)

        self.start()

    def write(self, label, data, timestamp=None, clockstamp=None, columns=None):
        if self.kill_flag.value:
            return

        timestamp = timestamp or now()
        clockstamp = clockstamp or now2()
        self.buffer.put([label, data, timestamp, clockstamp, columns])

    def run(self):
        try:
            with pd.HDFStore(self.data_file, mode='a') as f:
                f.put('code', pd.Series(self.session_obj.get_code_txt()))
                f.put('config', pd.Series(json.dumps(parse_config_dict(config.__dict__))))

            field_buffers = {} # categories items from self.buffer, then writes

            while True:
                if self.buffer.empty() and self.kill_flag.value:
                    break
                   
                try:
                    record = self.buffer.get(block=False)
                except queue.Empty:
                    continue
                
                label, data, ts, cs, columns = record

                if not isinstance(data, pd.DataFrame):
                    idx = np.atleast_1d(np.squeeze([ts]))
                    data = pd.DataFrame(data, columns=columns, index=idx)
                elif isinstance(data, pd.DataFrame):
                    data.set_index([[ts]*len(data)], inplace=True)

                data.loc[:, 'session'] = self.session_id
                data.loc[:, 'clockstamp'] = cs

                # add to label-specific buffer
                if label in field_buffers:
                    field_buffers[label].append(data)
                else:
                    field_buffers[label] = [data]
                
                # flush buffers as indicated
                field_buffer = field_buffers[label]
                if len(field_buffer) >= self.buffer_size:
                    self.flush_buffer(field_buffers, label)
            # end main loop

            # final write
            for label, field_buffer in field_buffers.items():
                self.flush_buffer(field_buffers, label)
                
        except Exception as e:
            self.error_queue.put(f'Saver: {str(e)}')
            
    def flush_buffer(self, field_buffers, label):
        field_buffer = field_buffers[label]

        if len(field_buffer) == 0:
            return

        to_write = pd.concat(field_buffer)
       
        try:
            with pd.HDFStore(self.data_file, mode='a') as f:
                f.append(label, to_write,
                              index=False,
                              data_columns=['session','clockstamp'],
                              complevel=0)
            field_buffers[label] = []
        except Exception as e:
            self.error_queue.put(f'Saver: {str(e)}')
                    
    def end(self):
        self.kill_flag.value = True

