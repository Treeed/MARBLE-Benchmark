import time

from benchmark.tasks.MUSDB18.MUSDB18_dataset import FixedSourcesTrackFolderDataset, aug_from_str

dataset_kwargs = {'root': 'data/musdb18_mp4', 'target_file': 'vocals', 'sample_rate': 24000}

source_augmentations = aug_from_str(["gain", "channelswap"])


dataloader = FixedSourcesTrackFolderDataset(split='train', seq_duration=6, source_augmentations=source_augmentations, random_chunks=True, random_track_mix=True, **dataset_kwargs)

start = time.time()
for num in range:
    tt = dataloader[num]
print(time.time()-start)