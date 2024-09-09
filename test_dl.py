import time
import torch

from benchmark.tasks.MUSDB18.MUSDB18_dataset import FixedSourcesTrackFolderDataset, aug_from_str

dataset_kwargs = {'root': 'data/musdb18_mp3', 'target_file': 'vocals.mp3', 'sample_rate': 24000}

source_augmentations = aug_from_str(["gain", "channelswap"])


dataloader = FixedSourcesTrackFolderDataset(split='train', seq_duration=6, source_augmentations=source_augmentations, random_chunks=True, random_track_mix=True, **dataset_kwargs)

start = time.time()
for num in range(64):
    tt = dataloader[num]
    assert tt[0].shape == (2, 24000*6) and tt[1].shape == (2, 24000*6)
    assert torch.sum(torch.abs(tt[0])) > 1 and torch.sum(torch.abs(tt[1])) > 1
print(time.time()-start)