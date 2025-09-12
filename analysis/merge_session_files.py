##
import pandas as pd
import os

to_merge = [
            '/Users/bdd/Desktop/drive-download-20250730T185854Z-1-001/2025-07-30_07-57-13_subject-b004.h5',
            '/Users/bdd/Desktop/drive-download-20250730T185854Z-1-001/2025-07-30_09-52-22_subject-b004.h5',
        ]
to_merge = sorted(to_merge)

out_name = '2025-07-30_merge_subject-b004'
out_path = f'/Users/bdd/Desktop/{out_name}.h5'

## merge

last_eeg_end_time = None

for m in to_merge:
    with pd.HDFStore(m, 'r') as hin, pd.HDFStore(out_path, 'a') as hout:
        for key in hin:
            print(key)
            if key == '/markers':
                min_itemsize = dict(text=500)
            else:
                min_itemsize = None

            if key == '/eeg':
                if last_eeg_end_time is not None:
                    Ts = 1/500
                    next_eeg_start_time = hin['/eeg'].index.values[0]
                    bridge_time = np.arange(last_eeg_end_time+Ts, next_eeg_start_time, Ts)
                    ex = hin['/eeg'].iloc[:5]
                    bridge_eeg = pd.DataFrame(index=bridge_time, columns=ex.columns, dtype=np.float64)
                    hout.append(key, bridge_eeg)
                else:
                    last_eeg_end_time = hin['/eeg'].index.values[-1]

            hout.append(key, hin[key], min_itemsize=min_itemsize)
            


## confirm
with pd.HDFStore(out_path, 'r') as h:
    for key in h:
        print(key, h[key].shape)
    pl.plot(h.eeg.index.values)

##
