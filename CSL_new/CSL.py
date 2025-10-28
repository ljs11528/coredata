import argparse
import os
import pytorch_lightning as pl 
import yaml
import torch
from dataset.semi import SemiDataset
from model.model_helper import ModelBuilder
from train.semi_supervised_train import SemiModule
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from supervised import find_latest_checkpoint

def main():   
    parser = argparse.ArgumentParser(description='Separating Optimization for Reliable Prediction\
                                      in Semi-supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--labeled_id_path', type=str, required=True)
    parser.add_argument('--unlabeled_id_path', type=str, required=True)
    parser.add_argument('--val_id_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = TensorBoardLogger(save_dir=args.save_path, version=1, name="logs")

    pl.seed_everything(42, workers=True)

    print(f"[pid {os.getpid()}] RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

    model = ModelBuilder(cfg['model'])
    
    trainset_u = SemiDataset(**{**cfg['dataset'], 'mode': 'train_u', 'id_path': args.unlabeled_id_path})
    trainset_l = SemiDataset(**{**cfg['dataset'], 'mode': 'train_l', 'id_path': args.labeled_id_path, 'nsample': len(trainset_u.ids)})
    valset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', args.val_id_path)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
    valloader =  DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, drop_last=False)

    train_loaders = {
        'labeled': trainloader_l,
        'unlabeled': trainloader_u,
        'mixed': trainloader_u,
    }

    batch_iters = max(len(trainloader_l), len(trainloader_u)) // cfg['gpu_num']
    total_iters = batch_iters * cfg['epochs']
    train_module = SemiModule(**{
        **cfg['train'],
        'model': model,
        'save_path': args.save_path, 
        'batch_iters': batch_iters, 
        'total_iters':total_iters,
        'nclass': cfg['nclass'] 
        })
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],     
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_false",
        benchmark=True,
        logger=logger,
        precision="bf16-mixed",
        sync_batchnorm=True,
        accumulate_grad_batches=cfg['accumulate_grad_batches'],
        enable_checkpointing=True,
        log_every_n_steps= batch_iters // 32,
    )

    checkpoint_path = find_latest_checkpoint(os.path.join(args.save_path, "checkpoints"))
    if trainer.is_global_zero and checkpoint_path != None:
        print("load checkpoint : ",checkpoint_path)

    trainer.fit(train_module, train_dataloaders=train_loaders, val_dataloaders=valloader, ckpt_path=checkpoint_path)



if __name__ == '__main__':
    main()