now=$(date +"%Y%m%d_%H%M%S")

# 你想用的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 禁用容器里常出问题的 NCCL 特性
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0   # 如果容器不是 eth0，请用 ip a 查真实网卡名

# torch.distributed 必需变量
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29510  # 换成没被占用的端口

dataset='pascal'
method='CSL'
exp='r101'
split='1_4'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
val_id_path=splits/$dataset/val.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

# 用 torchrun 启动，4 卡
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT $method.py \
    --config=$config \
    --labeled_id_path=$labeled_id_path \
    --unlabeled_id_path=$unlabeled_id_path \
    --val_id_path=$val_id_path \
    --save_path=$save_path
