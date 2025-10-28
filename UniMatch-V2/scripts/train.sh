#!/bin/bash

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'ade20k', 'coco']
# method: ['unimatch_v2', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='pascal'
method='unimatch_v2'
exp='dinov2_base'
split='366'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

# python -m torch.distributed.launch \
#     --nproc_per_node=$1 \
#     --master_addr=localhost \
#     --master_port=$2 \
#     $method.py \
#     --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
#     --save-path $save_path --port $2 2>&1 | tee $save_path/out.log

CUDA_VISIBLE_DEVICES=1,2 

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=12345 \
    $method.py \
    --config=configs/pascal.yaml \
    --labeled-id-path splits/pascal/366/labeled.txt \
    --unlabeled-id-path splits/pascal/366/unlabeled.txt \
    --save-path exp/pascal/unimatch_v2/dinov2_base/366 \
    --port 12345 2>&1 | tee exp/pascal/unimatch_v2/dinov2_base/366/out.log

