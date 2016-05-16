#!/usr/bin/env sh

set -x
set -e

UNK=$1
OPT=$2

for CUTOFF in 0 30 100 300 1000; do
  for NAME in pattern; do
    DATASET=${NAME}_${UNK}_cutoff${CUTOFF} &
    bin/convert_dataset.lua -i dataset/pattern -t $NAME -c $CUTOFF --unk $UNK -o dataset/${DATASET}${OPT} $OPT &
  done
done
