##
from gtts import gTTS
import os

import fus_anes.config as config

interval = config.verbal_instruction_interval
n_create = config.verbal_instructions_n_prepare
name = config.name.lower()
command = config.verbal_instruction_command
out_path = os.path.join(config.verbal_instructions_path, name)
slow = False

if __name__ == '__main__':
    os.makedirs(out_path, exist_ok=True)
    
    text = f'{name}, {command}...'
    tts = gTTS(text=text, lang='en', slow=slow)
    tts.save(os.path.join(out_path, f"{name}.mp3"))

    for i in range(1, n_create+1):
        text = f'{name}, {command}...  {i}.'
        tts = gTTS(text=text, lang='en', slow=slow)
        tts.save(os.path.join(out_path, f"{name}_{i}.mp3"))


##
