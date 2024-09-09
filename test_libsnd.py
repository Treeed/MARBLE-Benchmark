import contextlib
import random
from pathlib import Path
import numpy as np
import re

import soundfile

filepath = Path("data/musdb18_mp3/train/")

def load_and_seek(*args, start, framesize, **kwargs):
    start_block_start = max(start-25000, 0)
    with contextlib.redirect_stdout(None):
        dat = soundfile.read(*args, start=start_block_start, **kwargs)[0]
    return dat[start-start_block_start:].T



for path in filepath.iterdir():
    for stem in ["bass.mp3", "drums.mp3", "other.mp3", "vocals.mp3", "mixture.mp3"]:

        info = soundfile.info(path/stem)
        end = int(random.random()*info.duration*24000)
        start = int(random.random()*end)
        framesize = int(re.search("framesize +: (\d+)", soundfile.info(path/stem).extra_info).group(1))
        data = load_and_seek(path / stem, start=start, stop=end, framesize=framesize)

        alt = soundfile.read(path/stem)[0]
        alt = alt[start:end].T

        diff = data-alt
        if np.abs(diff).max() > 0.001:
            print(np.abs(diff).max())

        print("hi")