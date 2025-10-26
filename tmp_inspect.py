from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = '/202431510182/LLM/realman/lerobot_v30_out_v3_videos2'

ds = LeRobotDataset('lerobot_v30_out_v3_videos2', root=root)
print('meta keys:', list(ds.meta.features.keys()))
item = ds[0]
print('item keys:', list(item.keys()))
for k, v in item.items():
    if hasattr(v, 'shape'):
        print(k, getattr(v, 'shape'))
print('done')
