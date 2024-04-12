from dotenv import load_dotenv

load_dotenv('../.env', override=True)  # load API keys into

import os
from tqdm import tqdm
from transformers.utils.hub import cached_file

print(os.getenv("HF_HOME"))
cfiles = [i for i in os.listdir('../configs/') if (not i.startswith('__') and ("config_710003" not in i))]
cIDs = [int(idx.split("_")[1].split('.')[0]) for idx in cfiles]

pbar = tqdm(cIDs)
for config_id in pbar:
    if not str(config_id).startswith('-'):
        pbar.set_description(f'Downloading models for config {config_id}')
        try:
            cached_file(f"nightdude/config_{config_id}", 'adapter_config.json')
            cached_file(f"nightdude/config_{config_id}", 'adapter_model.bin')
        except Exception as e:
            print(e)