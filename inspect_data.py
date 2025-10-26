import pyarrow.parquet as pq
from pathlib import Path
path = Path('/202431510182/LLM/realman/lerobot_v30_out_v3_videos2/data/chunk-000/data.parquet')
table = pq.read_table(path)
with open('data_info.txt', 'w') as f:
    f.write('columns: '+ str(table.column_names)+'\n')
    f.write('nrows: '+ str(table.num_rows)+'\n')
    f.write(str(table.schema)+'\n')
