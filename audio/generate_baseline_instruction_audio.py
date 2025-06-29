##
from gtts import gTTS
import os

out_path = '/Users/bdd/code/fus_anes/media/baseline_audio'

if __name__ == '__main__':
    os.makedirs(out_path, exist_ok=True)

    text = f'Now keep your eyes closed.'
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(os.path.join(out_path, f"closed.mp3"))

    text = f'Now keep your eyes open.'
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(os.path.join(out_path, f"open.mp3"))

##
