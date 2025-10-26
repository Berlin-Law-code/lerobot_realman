import pandas as pd
from pathlib import Path
root = Path('/202431510182/LLM/realman/lerobot_v30_out_v3_videos2/meta')
out = root / 'episodes_head.txt'
df = pd.read_json(root / 'episodes.jsonl', lines=True)
with open(out, 'w') as f:
    f.write(str(df.columns.tolist()) + '\n')
    f.write(df.head(1).to_json())
