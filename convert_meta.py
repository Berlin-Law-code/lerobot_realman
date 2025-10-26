import pandas as pd
from pathlib import Path
root = Path('/202431510182/LLM/realman/lerobot_v30_out_v3_videos2/meta')

# Episodes
episodes_jsonl = root / 'episodes.jsonl'
if episodes_jsonl.exists():
    df = pd.read_json(episodes_jsonl, lines=True)
    out_dir = root / 'episodes'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'chunk-000'
    out_path.mkdir(exist_ok=True)
    (out_path / 'file-000.parquet').unlink(missing_ok=True)
    df.to_parquet(out_path / 'file-000.parquet')

# Episodes stats optional
stats_jsonl = root / 'episodes_stats.jsonl'
if stats_jsonl.exists():
    df = pd.read_json(stats_jsonl, lines=True)
    out_dir = root / 'episodes_stats'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'chunk-000'
    out_path.mkdir(exist_ok=True)
    (out_path / 'file-000.parquet').unlink(missing_ok=True)
    df.to_parquet(out_path / 'file-000.parquet')
