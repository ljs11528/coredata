import argparse
import os
import pytorch_lightning as pl 
import yaml
from dataset.semi import SemiDataset
from model.model_helper import ModelBuilder
from train.supervised_train import SupervisedModule
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    return None


def main():
    parser = argparse.ArgumentParser(description='Separating Optimization for Reliable Prediction\
                                      in Semi-supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--labeled_id_path', type=str, required=True)
    parser.add_argument('--val_id_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = TensorBoardLogger(save_dir=args.save_path, version=1, name="logs")

    pl.seed_everything(42, workers=True)
    
    model = ModelBuilder(cfg['model'])
    
    trainset_l = SemiDataset(**{**cfg['dataset'], 'mode': 'train_l', 'id_path': args.labeled_id_path})
    valset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', args.val_id_path)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
    
    valloader =  DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, drop_last=False)


    batch_iters = len(trainloader_l) // cfg['gpu_num']
    total_iters = batch_iters * cfg['epochs']
    cfg['train'].pop('alpha', None)
    train_module = SupervisedModule(**{
        **cfg['train'],
        'model': model,
        'save_path': args.save_path, 
        'total_iters':total_iters
        })
    
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],     
        accelerator='auto',
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

    trainer.fit(train_module, train_dataloaders=trainloader_l, val_dataloaders=valloader, ckpt_path=checkpoint_path)

if __name__ == '__main__':
    main()