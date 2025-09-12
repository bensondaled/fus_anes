from fus_anes.constants import *

# Flags
THREADS_ONLY = False
TESTING = False
SIM_DATA = False

# Subject
subject_id = 'b008'
age = 29
weight = 65.9
height = 171.0
sex = 'f'
name = 'kate'

# Paths
data_path = f'C:/data_burst/{subject_id}'
logging_file = 'C:/data_burst/log.log'
loading_img_path = 'media/propofol.png'
post_test_graphics_path = 'post_tests/graphics/'
baseline_audio_path = 'media/baseline_audio'
oddball_audio_path = 'media/oddball_audio/'
chirp_audio_path = 'media/chirp_audio/chirp.wav'
chirp_white_audio_path = 'media/chirp_audio/chirp_white.wav'

# Verbal instructions
verbal_instructions_path = data_path
squeeze_path = f'{verbal_instructions_path}/squeeze_audio'
verbal_instruction_interval = (0.750, 4.5) # secs: (min, max)
verbal_instructions_n_prepare = 120
squeeze_beep_f = 440
squeeze_beep_dur = 0.150
squeeze_beep_delay = [150, 800] # ms
use_squeeze_beep = False

# EEG acquisition
fs = 500 # Hz
chan_reference = passive_reference_electrode_idx 
eeg_memory_length = 5000 # samples
read_buffer_length = 25 # samples
eeg_key = MONTAGE
n_channels = len(eeg_key)
eeg_init_chan_selects = [passive_eeg_frontal, passive_eeg_posterior]
max_freq = 40 # of physiologic interest, Hz
spect_update_interval = 5000 # samples (use multiple of fs and <eeg_memory_length)
spect_memory_length = 60*60*3 # seconds
spect_freq_downsample = 5
save_buffer_length = 20000 # samples, mult of read_buf_len
eeg_n_timefields = 5 # 3 for the now() output, 2 for the added inlet/offset

# EEG display
n_live_chan = 2
raw_eeg_display_dur = 4.0 # secs (<= eeg_memory_length in time)
eeg_lopass = 70
eeg_hipass = 0.1
eeg_notch = 60
n_spect_time_selections = 3
n_spect_freq_selections = 2
spect_freq_defaults = [(0.5, 4), (9, 15)]
cmap = 'rainbow'
eeg_baseline_gain = 1
eeg_gain_zoom_factor = 10000 / eeg_baseline_gain # high: finer
spect_log = True
eeg_special_filters = { 17: dict(lo=20, hi=0.1, notch=eeg_notch, gain=0.000005), # gripswitch
                        16: dict(lo=249, hi=60, notch=eeg_notch, gain=0.001), # ssep
                        15: dict(lo=eeg_lopass, hi=eeg_hipass, notch=eeg_notch, gain=0.1), # ecg
                      }

# Misc display
timeline_duration = 20*60 # secs, default time range to show
timeline_advance = 2.5*60 # secs, how much to jump in advance when you hit the end of timeline

# Camera
cam_frame_size = [480, 640, 3]
cam_resize = 1.0
cam_file_duration = 10*60 # secs
fourcc = 'XVID'
mov_ext = 'avi' # xvid=avi, -1=mp4

# Sound
audio_backend = 'ptb'
audio_in_ch_out_ch = [8, 4] # mic, speaker, use audio_util.probe_audio_devices
audio_playback_delay = 0.100
audio_playback_fs = 44100
audio_stream_chunk = 8192
audio_save_chunk = audio_stream_chunk * 50
audio_hdf_resize = audio_save_chunk * 10
n_audio_display = 3.0 # secs

# TCI
drug = 'propofol'
pump_port = 'COM3' # Device manager: "Prolific USB-to-Serial..."
pump_rate_min = 0.002
tci_use_prior_session = True
tci_sim_duration = 8*60 # secs
tci_minval = 5
tci_display_target = (0.5, 2.0)
syringe_diam = 26.7 #mm, BD 50/60cc syringe
bolus_rate = 25 # ml/min
max_bolus = 300 # mg
max_rate = 25 # ml/min, 28.4 is manufacturer max
min_rate = 0.005 # ml/min, 0.002 is manufacturer min
hold_level_duration = 15*60 # secs
drug_mg_ml = 10.0 # 10mg/ml for propofol
goto_target_step_size = 15.0

# Oddball
oddball_deviant_ratio = 0.15
oddball_isi_ms = [630, 780]
oddball_n_tones = 250
oddball_n_standard_start = 15

# Chirp
chirp_n_tones = 100 # will have 10 additional chirp at start
chirp_isi_ms = (1220, 1520)
chirp_ctl_rate = 0.1
chirp_n_start = 10
