#!/bin/bash

BINARY="./cmake-build-release/benchMonteCarlo"

for i in 1 4 8 12
do
    OMP_NUM_THREADS=$i "$BINARY" \
        --benchmark_out="threads_$i.json" \
        --benchmark_out_format=json
done
