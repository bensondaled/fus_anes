# conda activate psychopy

from psychopy import visual, core, event, gui
from psychopy.hardware import keyboard
import random
import csv
import os
import json
import numpy as np
from datetime import datetime

import fus_anes.config as config

data_file = os.path.join(config.data_path, f'{config.subject_id}_post_tests.txt')
red_path = os.path.join(config.post_test_graphics_path, 'red.jpeg')
symbol_path = os.path.join(config.post_test_graphics_path, 'symbols')
num_path = os.path.join(config.post_test_graphics_path, 'nums')

win = visual.Window(fullscr=False, color='black', units='norm')
text = visual.TextStim(win, text='', color='white', height=0.07, wrapWidth=1.5)
kb = keyboard.Keyboard()
kb.clearEvents()

def log(task, data):
    with open(data_file, 'a') as f:
        row = [datetime.now().isoformat(),
               task,
               data]
        row = json.dumps(row)
        row = f'{row}\n'
        f.write(row)

def show_msg(message, wait_keys=['space']):
    text.text = message
    text.draw()
    win.flip()
    event.waitKeys(keyList=wait_keys)




# === Psychomotor Vigilance Task (Visual) ===
red_box = visual.ImageStim(win, image=red_path,
								size=(0.4,0.4),
								pos=[0, 0],
                                units='norm')
yellow_counter = visual.TextStim(win, text='', color='yellow', height=0.1)

total_trials = 5
min_interval = 2
max_interval = 10

for trial in range(total_trials):
    red_box.draw()
    win.flip()

    start_time = kb.clock.getTime()

    kb.clearEvents()
    kb.clock.reset()
    start_delay = random.uniform(min_interval, max_interval)
    while kb.clock.getTime() - start_time < start_delay:
        keys = kb.getKeys(['space'], waitRelease=True)
        if keys:
            response = keys[0]
            rt = response.rt
            log('pvt', dict(delay=start_delay, rt=rt, note='early', displayed_rt_ms=''))

    kb.clearEvents()
    kb.clock.reset()
    # Update yellow counter every frame until response
    while True:
        now = kb.clock.getTime()
        ms = int(now * 1000)
        yellow_counter.text = f"{ms} ms"

        red_box.draw()
        yellow_counter.draw()
        win.flip()
        
        keys = kb.getKeys(['space'], waitRelease=True)
        if keys:
            response = keys[0]
            rt = response.rt
            break

    log('pvt', dict(delay=start_delay, rt=rt, displayed_rt_ms=ms, note=''))
    core.wait(1.0)
    win.flip()
    core.wait(1.0)




# === Digit Symbol Substitution Task (DSST) ===
# Visual key map setup
symbol_names = sorted(os.listdir(symbol_path))
np.random.shuffle(symbol_names)
num_names = sorted(os.listdir(num_path))
              
allowed_responses = [str(i) for i in range(1,10)]
key_images = []
ypos_s = 0.7
ypos_n = 0.5
size=(0.15, 0.15)
x_positions = np.linspace(-0.8, 0.8, len(num_names))
for i,(sn,nn) in enumerate(zip(symbol_names, num_names)):
    sym_img = visual.ImageStim(win, image=os.path.join(symbol_path, sn),
                                    size=size,
                                    pos=[x_positions[i], ypos_s])
    num_img = visual.ImageStim(win, image=os.path.join(num_path, nn),
                                    size=size,
                                    pos=[x_positions[i], ypos_n])
    key_images.append([sym_img, num_img])

log('dsst', dict(sym=symbol_names, num=num_names))
for i in range(10):
    ans = np.random.choice(symbol_names)
    
    # show key for a sec
    for im0,im1 in key_images:
        im0.draw()
        im1.draw()
    win.flip()
    core.wait(1.0)
    
    # show question with it
    for im0,im1 in key_images:
        im0.draw()
        im1.draw()
    central_img = visual.ImageStim(win, image=os.path.join(symbol_path, ans), size=(0.25, 0.25), pos=(0, -0.2))
    central_img.draw()
    win.flip()

    kb.clock.reset()
    key = None
    while not key:
        key = kb.getKeys(allowed_responses, waitRelease=True)
        core.wait(0.01)
    key = key[0]
    log('dsst', dict(sym=ans, key=key.name, rt=key.rt))
    
    

win.close()
core.quit()
