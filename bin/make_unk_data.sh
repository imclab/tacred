#!/usr/bin/env sh

set -x
set -e

UNK=$1
OPT=$2

PREFIX=train+pattern

for CUTOFF in 0 30 100 300 1000; do
  DATASET=${PREFIX}_${UNK}_cutoff${CUTOFF}
  bin/convert_dataset.lua -i dataset/${PREFIX} -c $CUTOFF --unk $UNK -o dataset/${DATASET}${OPT} $OPT &
done
