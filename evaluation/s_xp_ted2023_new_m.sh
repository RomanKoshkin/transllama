#!/bin/bash

# load the merged (base+LoRA) model in 16-bit instead of the base model in 4-bit to LoRAs separately.
# this agent (s2tt_agent_NEW.py) is different from s2tt_agent.py in that it accepts min_read_time as a parameter
for MIN_READ_TIME in 1.9; do
    for MIN_LAG in 1; do
        python se.py \
            --source SOURCES/src_ted_new_tst_100.de \
            --target OFFLINE_TARGETS/tgt_ted_new_tst_100.de \
            --agent s2tt_agent_NEWm.py \
            --config_id 811357000 \
            --wait_k 1 \
            --device_id -1 \
            --source-segment-size 200 \
            --asr_model whisper-small.en \
            --min_lag $MIN_LAG \
            --min_read_time $MIN_READ_TIME \
            --end-index 102
    done
done