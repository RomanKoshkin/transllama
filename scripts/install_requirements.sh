#!/bin/bash

pip install -U git+https://github.com/huggingface/transformers.git@e03a9cc
pip install peft==0.8.2
pip install python-dotenv
pip install simalign==0.3
pip install numpy==1.23.5
pip install nltk==3.8.1
pip install numba==0.57.1
pip install bitsandbytes==0.39.0
pip install janome==0.5.0
pip install termcolor==2.4.0
pip install datasets==2.18.0
pip install wandb==0.15.4
cd ../SimulEval & pip install -e . & cd ../scripts
pip install einops==0.6.1
cd ../bleurt & pip install -e . & cd ../scripts
pip install openai==0.27.10
pip install tiktoken==0.5.2
pip install bitsandbytes==0.39.0
