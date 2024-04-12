#!/bin/bash

for WAIT_K in 1 2 3 4 5 6 7 8; do
    for SEG_SZ in 200; do
        for MIN_LAG in 1; do
            python se.py \
                --source SOURCES/src_ted_new_tst_100.de \
                --target OFFLINE_TARGETS/tgt_ted_new_tst_100.de \
                --agent s2tt_agent.py \
                --config_id 810357 \
                --wait_k $WAIT_K \
                --device_id $1 \
                --source-segment-size $SEG_SZ \
                --asr_model whisper-large-v2 \
                --min_lag $MIN_LAG
        done
    done
done