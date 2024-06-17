#!/bin/bash
## 811357
for MIN_READ_TIME in 1.9; do
    for MIN_LAG in 1; do
        python se.py \
            --source /home/roman/CODE/gremlin/evaluation/SOURCES/fleurs_en_de_102 \
            --target /home/roman/CODE/gremlin/evaluation/OFFLINE_TARGETS/fleurs_en_de_102.txt \
            --agent s2tt_agent_NEW.py \
            --config_id 811357 \
            --wait_k 1 \
            --device_id 0 \
            --source-segment-size 200 \
            --asr_model whisper-small.en \
            --min_lag $MIN_LAG \
            --min_read_time $MIN_READ_TIME \
            --end-index 102
    done
done