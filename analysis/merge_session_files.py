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

##
for m in to_merge:
    with pd.HDFStore(m, 'r') as hin, pd.HDFStore(out_path, 'a') as hout:
        for key in hin:
            hout.append(key, hin[key])
            


##

##
