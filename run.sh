 #! /bin/bash
dataset='laptop'

model_list=(base large roberta)
for model in ${model_list[@]}
do
    python train.py \
        --dataset $dataset \
        --model_name $model \
done