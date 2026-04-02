python3 -c --from datasets import load_dataset --import numpy as np --ds = load_dataset{'Sri-Ram-A/pnp1', split='train'} --actions = np.array{ds['action'][:100]} --print{'Action min:', actions.min{axis=0}} --print{'Action max:', actions.max{axis=0}} --print{'Action mean:', actions.mean{axis=0}}

# rad-degee
from datasets import load_dataset
import numpy as np
ds = load_dataset('Sri-Ram-A/pnp1', split='train')
actions = np.array(ds['action'][:100])
print('Action min:', actions.min(axis=0))
print('Action max:', actions.max(axis=0))
print('Action mean:', actions.mean(axis=0))
"
README.md: 3.95kB [00:00, 19.7MB/s]
data/chunk-000/file-000.parquet: 100%|███████| 910k/910k [00:04<00:00, 227kB/s]
Generating train split: 100%|█| 30441/30441 [00:00<00:00, 853131.51 examples/s]
Action min: [ 1.34288857e-03 -4.44421396e-02  1.97033150e-05 -2.31653780e-01
  2.32106773e-03 -9.76095907e-05]
Action max: [0.66903448 0.02866589 0.20299269 0.02781744 0.00273629 1.50007021]
Action mean: [ 0.32831938 -0.01949339  0.01771948  0.00138886  0.00249239  0.91497145]

import json
from huggingface_hub import hf_hub_download

f = hf_hub_download('Sri-Ram-A/pnp1', 'meta/info.json', repo_type='dataset')
info = json.load(open(f))
print(info.get('features', {}).get('action', {}))
{'dtype': 'float32', 'shape': [6], 'names': ['motor_1', 'motor_2', 'motor_3', 'motor_4', 'motor_5', 'motor_6'], 'fps': 30}