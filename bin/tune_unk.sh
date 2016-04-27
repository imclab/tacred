#!/usr/bin/env sh

set -x
set -e

UNK=$1
PCOR=$2
GPU=$3

for CUTOFF in 0 30 100 300 1000; do
  DATASET=${UNK}_cutoff${CUTOFF}
  SAVE=${UNK}_cutoff${CUTOFF}_pcor${PCOR}
  bin/train.lua --dataset dataset/${DATASET} --gpu $GPU --save saves/${SAVE} --p_corrupt $PCOR &
done
