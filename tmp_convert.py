import pandas as pd
from pathlib import Path
root = Path('/202431510182/LLM/realman/lerobot_v30_out_v3_videos2/meta')
jsonl = root / 'tasks.jsonl'
parquet = root / 'tasks.parquet'
log = root / 'convert.log'
with open(log, 'w') as f:
    f.write('jsonl exists? %s\n' % jsonl.exists())
    if jsonl.exists():
        df = pd.read_json(jsonl, lines=True)
        f.write('rows: %d\n' % len(df))
        df.to_parquet(parquet)
        f.write('done\n')
    else:
        f.write('missing jsonl\n')
