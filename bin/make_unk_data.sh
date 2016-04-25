#!/usr/bin/env sh

set -x
set -e

UNK=$1
DATASET=${UNK}_cutoff${CUTOFF}

for CUTOFF in 0 30 100 300 1000; do
  bin/convert_dataset.lua -c $CUTOFF --unk $UNK -o dataset/${DATASET} &
done
