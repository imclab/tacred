#!/usr/bin/env sh

set -x
set -e

UNK=$1
PCOR=$2
GPU=$3
OPT=$4

PREFIX=train+pattern

for CUTOFF in 0 30 100 300 1000; do
  DATASET=${PREFIX}_${UNK}_cutoff${CUTOFF}$OPT
  SAVE=${PREFIX}_${UNK}_cutoff${CUTOFF}_pcor${PCOR}$OPT
  bin/train.lua --dataset dataset/${DATASET} --gpu $GPU --save saves/${SAVE} --p_corrupt $PCOR &
done