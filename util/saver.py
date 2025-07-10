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

from fus_anes.util import now
import fus_anes.config as config

def save(label, data, buffer, time_data=None, columns=None):
    if time_data is None:
        lslstamp, timestamp, perfstamp = now()
    else:
        lslstamp, timestamp, perfstamp = time_data
    buffer.put([label, data, lslstamp, timestamp, perfstamp, columns])

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

        self.buffer = mp.Queue() # all items to save go through this
        self.running = mp.Value('b', False)
        self.start()

    def run(self):
        self.running.value = True

        try:
            with pd.HDFStore(self.data_file, mode='a') as f:
                f.put('code', pd.Series(self.session_obj.get_code_txt()))
                f.put('config', pd.Series(json.dumps(parse_config_dict(config.__dict__))))

            field_buffers = {} # categories items from self.buffer, then writes

            while True:
                try:
                    record = self.buffer.get(block=True, timeout=None)

                    if record is None: # sentinel to end
                        break
                    
                    label, data, ls, ts, ps, columns = record

                    if not isinstance(data, pd.DataFrame):
                        idx = np.atleast_1d(np.squeeze([ls]))
                        data = pd.DataFrame(data, columns=columns, index=idx)
                    elif isinstance(data, pd.DataFrame):
                        data.set_index([[ls]*len(data)], inplace=True)

                    data.loc[:, 'session'] = self.session_id
                    data.loc[:, 'pc_stamp'] = ps
                    data.loc[:, 'time_stamp'] = ts

                    # add to label-specific buffer
                    if label in field_buffers:
                        field_buffers[label].append(data)
                    else:
                        field_buffers[label] = [data]
                    
                    # flush buffers as indicated
                    field_buffer = field_buffers[label]
                    if len(field_buffer) >= self.buffer_size:
                        self.flush_buffer(field_buffers, label)
                except Exception as e:
                    self.error_queue.put(f'Saver: {str(e)}')
            # end main loop

            # final write
            for label, field_buffer in field_buffers.items():
                self.flush_buffer(field_buffers, label)
                
        except Exception as e:
            self.error_queue.put(f'Saver: {str(e)}')

        self.running.value = False
            
    def flush_buffer(self, field_buffers, label):
        field_buffer = field_buffers[label]

        if len(field_buffer) == 0:
            return

        to_write = pd.concat(field_buffer)
       
        try:
            with pd.HDFStore(self.data_file, mode='a') as f:
                #print(label, to_write.dtypes)
                f.append(label, to_write,
                              index=False,
                              data_columns=['session','time_stamp'],
                              complevel=0)
                #print(f[label].dtypes)
            field_buffers[label] = []
        except Exception as e:
            self.error_queue.put(f'Saver buffer flush: {str(e)}')
                    
    def end(self):
        self.buffer.put(None) # sentinel flag to end
        while self.running.value:
            time.sleep(0.100)

