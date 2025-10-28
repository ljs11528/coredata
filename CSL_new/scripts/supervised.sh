now=$(date +"%Y%m%d_%H%M%S")
export CUDA_VISIBLE_DEVICES=0,1,2,3

# modify these augments if you want to try other datasets or splits
# dataset: ['pascal', 'cityscapes']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits

dataset='pascal'
method='supervised'
exp='r101'
split='1_4'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
val_id_path=splits/$dataset/val.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path
python $method.py --config=$config --labeled_id_path $labeled_id_path \
    --val_id_path $val_id_path --save_path $save_path 