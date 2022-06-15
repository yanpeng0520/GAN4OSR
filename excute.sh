#!/bin/bash

gpu=3
niter=10
nz=256
gen_arch='small'
arch='LeNet_plus_plus'
dataset='openset_mnist_emnist'
dataroot='/local/scratch/pyan'
dataroot='/tmp'
batchSize=64
methods=('SoftMax' 'BG' 'Cross_KU' 'Cross_noise' 'Cross_adv' 'Cross_gan')

# training for different approaches
START_train=$(date +%s)
for m in ${methods[@]}
do
    echo Training $m
    START_train_m=$(date +%s)
    mkdir -p "./output"
    nice -n 19 python3 train.py --cuda --dataroot $dataroot --dataset $dataset --batchSize $batchSize --niter $niter --gpu $gpu --nz $nz -a entropic -fm ${m} --arch $arch --gen_arch $gen_arch | tee "./output/${m}".txt
    END_train_m=$(date +%s)
    train_time=$(( $END_train_m - $START_train_m ))
    printf 'Training took %dh:%dm:%ds\n' $((train_time/3600)) $((train_time%3600/60)) $((train_time%60))
    echo ------------------------------------------------
done

END_train=$(date +%s)
train_time=$(( $END_train - $START_train ))
echo "Training took $DIFF seconds"
printf 'Training took %dh:%dm:%ds\n' $((train_time/3600)) $((train_time%3600/60)) $((train_time%60))

#evaluation
START_eval=$(date +%s)

nice -n 19 python3 evaluate.py --cuda --gpu $gpu --arch $arch --dataset $dataset --dataroot $dataroot

echo Evaluating
END_eval=$(date +%s)
eval_time=$(( $END_eval - $START_eval ))
echo "Evaluating took $DIFF seconds"
printf 'Evaluating took %dh:%dm:%ds\n' $((eval_time/3600)) $((eval_time%3600/60)) $((eval_time%60))

