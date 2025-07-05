from fus_anes.constants import *

# Flags
THREADS_ONLY = False
TESTING = False
SIM_DATA = False

# Subject
subject_id = 'test_subject'
age = 40
weight = 70.0
height = 180.0
sex = 'f'
name = 'sarah'

# Paths
data_path = 'C:/data_burst'
logging_file = 'logs/burst_log.log'
loading_img_path = 'media/propofol.png'

# Verbal instructions
baseline_audio_path = 'C:/code/fus_anes/media/baseline_audio'
verbal_instructions_path = data_path
verbal_instruction_interval = 0.5, (3.0, 1.5) # secs: min, (mean, sd)
verbal_instruction_command = 'squeeze'
verbal_instructions_n_prepare = 120

# EEG acquisition
fs = 500 # Hz
chan_reference = passive_reference_electrode_idx 
eeg_memory_length = 5000 # samples
read_buffer_length = 25 # samples
eeg_key = BCA_SL_32
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

# Misc display
timeline_duration = 20*60 # secs, default time range to show
timeline_advance = 2.5*60 # secs, how much to jump in advance when you hit the end of timeline

# Camera
cam_frame_size = [480, 640, 3]
cam_resize = 1
cam_file_duration = 10*60 # secs
fourcc = -1#'XVID'
mov_ext = 'mp4' # xvid=avi, -1=mp4

# Sound
audio_in_ch_out_ch = [2, 3]
audio_stream_chunk = 8192
audio_save_chunk = audio_stream_chunk * 50
audio_hdf_resize = audio_save_chunk * 10
audio_device_idx = 1 # use audio_probe script to identify mics
n_audio_display = 3.0 # secs

# TCI
drug = 'propofol'
pump_port = 'COM3' # Device manager: "Prolific USB-to-Serial..."
tci_use_prior_session = True
tci_sim_duration = 8*60 # secs
tci_minval = 10
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
oddball_audio_path = 'C:/code/fus_anes/media/oddball_audio/'
oddball_deviant_ratio = 0.20
oddball_isi_ms = 500
oddball_n_tones = 300
oddball_n_standard_start = 15

# Chirp
chirp_audio_path = 'C:/code/fus_anes/media/chirp_audio/chirp.wav'
chirp_white_audio_path = 'C:/code/fus_anes/media/chirp_audio/chirp_white.wav'
chirp_n_tones = 100 # will have 10 additional chirp at start
chirp_isi_ms = (1220, 1520)
chirp_ctl_rate = 0.1
chirp_n_start = 10
