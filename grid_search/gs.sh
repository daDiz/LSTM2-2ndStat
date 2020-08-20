#!/bin/bash

name=$1

for beta in 0.1 1 10 100 1000
do
    for dim in 64 128 256 512
    do
        for lr in 0.01 0.1 1.0 10.0
        do
            echo "beta$beta dim$dim lr$lr"
            . ../venv/bin/activate
            python train.py $beta $dim $lr $name >& train.log
            python predict.py $beta $dim $name >& predict.log
	    python gen_seq.py $beta $dim $name >& gen_seq.log
            deactivate

	    python hit.py >& hit_"$beta"_"$dim"_"$lr".txt
        done
    done
done
