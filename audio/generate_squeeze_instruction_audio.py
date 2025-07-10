##
from gtts import gTTS
import os

import fus_anes.config as config
from fus_anes.audio.audio_util import load_audio

interval = config.verbal_instruction_interval
n_create = config.verbal_instructions_n_prepare
name = config.name.lower()
out_path = os.path.join(config.verbal_instructions_path, name)
os.makedirs(out_path, exist_ok=True)
save_path = os.path.join(out_path, f'{name}.mp3')

command = 'squeeze at the beep'
slow = False

def make():
    if os.path.exists(save_path):
        return

    text = f'{name}, {command}...'
    tts = gTTS(text=text, lang='en', slow=slow)
    tts.save(save_path)

    # for numbered instructions:
    #for i in range(1, n_create+1):
    #    text = f'{name}, {command}...  {i}.'
    #    tts = gTTS(text=text, lang='en', slow=slow)
    #    tts.save(os.path.join(out_path, f"{name}_{i}.mp3"))


if __name__ == '__main__':
    make()
##
