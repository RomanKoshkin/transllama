#!/bin/bash

STEP=1
read -p "Number of runs:" LIM

for FROM in $(seq 0 $STEP $((LIM-STEP))); do
    python resample_new_ted.py ru &
    python se.py \
        --source resampled_src_new_ted_audio.txt \
        --target resampled_tgt_new_ted.txt \
        --agent s2tt_openai_agent.py \
        --wait_k 5 \
        --config_id -80219 \
        --source-segment-size 200 \
        --min_lag 1 \
        --asr_model whisper-large-v2 \
        --device_id 0
done