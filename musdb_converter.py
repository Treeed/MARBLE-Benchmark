import audiofile
import musdb
import pathlib
from joblib import Parallel, delayed


def process(split, track):
    pathlib.Path(f"data/musdb18_mp3/{split}/{track.name}/").mkdir(parents=True)
    print(track.name)
    for name, source in track.sources.items():
        audiofile.write(f"data/musdb18_mp3/{split}/{track.name}/{name}.mp3", source.audio.T, track.sample_rate)
    audiofile.write(f"data/musdb18_mp3/{split}/{track.name}/mixture.mp3", track.audio.T, track.sample_rate)


jobs = []

for split in ["test", "valid", "train"]:
    mus = musdb.DB(
        root="data/musdb18_mp4",
        is_wav=False,
        split=None if split == "test" else split,
        subsets="train" if split == "valid" else split,
        download=False,
        sample_rate=24000
    )
    for track in mus.tracks:
        jobs.append(delayed(process)(split, track))

Parallel(n_jobs=6)(jobs)