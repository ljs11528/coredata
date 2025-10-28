import os
import torch
from torch import nn
from torch import optim
from util.utils import AverageMeter, intersectionAndUnion
from util.ohem import ProbOhemCrossEntropy2d
from util.classes import CLASSES
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from util.loggingCallback import LoggingCallback


class SupervisedModule(pl.LightningModule):   
    def __init__(self, model, save_path, eval_mode, nclass, crop_size, lr, lr_multi, total_iters, name, criterion):
        super().__init__()
        self.model = model
        self.eval_mode = eval_mode
        self.nclass = nclass
        self.crop_size = crop_size
        self.lr = lr
        self.lr_multi = lr_multi
        self.total_iters = total_iters
        self.dataset = name
        self.save_path = save_path
        self.save_hyperparameters(ignore=['model'])
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        assert self.eval_mode in ['original', 'center_crop', 'sliding_window']
        if criterion['name'] == 'CELoss':
            self.criterion_l = nn.CrossEntropyLoss(**criterion['kwargs'])
        elif criterion['name'] == 'OHEM':
            self.criterion_l = ProbOhemCrossEntropy2d(**criterion['kwargs'])
        else:
            raise NotImplementedError(f"{criterion['name']} criterion is not implemented")
        
    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):  
        img, mask = batch  
        pred = self(img, False)  
        loss = self.criterion_l(pred, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, rank_zero_only=True)  
        return loss  
    
    def on_train_epoch_start(self):
        self.log('lr', self.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)  

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.scheduler.step()

    def validation_step(self, batch, batch_idx):  
        img, mask, id = batch  
        if self.eval_mode == "center_crop":
            pred, mask = self.center_crop_eval(img, mask)
        elif self.eval_mode == "sliding_window":
            pred = self.sliding_window_eval(img)
        else:
            pred = self(img, False).argmax(dim=1)
        intersection, union, target = \
            intersectionAndUnion(pred, mask, self.nclass, 255)
        self.intersection_meter.update(intersection)
        self.union_meter.update(union)

    def on_validation_epoch_start(self):
        self.intersection_meter.reset()
        self.union_meter.reset()

    def on_validation_epoch_end(self):
        intersection_gathered = self.all_gather(self.intersection_meter.sum).sum(dim=0)
        union_gathered = self.all_gather(self.union_meter.sum).sum(dim=0)
        iou_class =  intersection_gathered / (union_gathered + 1e-10) * 100.0
        mIOU = torch.mean(iou_class)
        
        for i, iou in enumerate(iou_class):
            self.log(f'val_iou_class_{i}_{CLASSES[self.dataset][i]}', iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True, batch_size=1)
            self.logging_callback.info(f'***** Evaluation ***** >>>> Class [{i} {CLASSES[self.dataset][i]}] IoU: {iou:.2f}')
        self.log("val_mIOU", mIOU, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, sync_dist=True, batch_size=1)
        self.logging_callback.info(f'***** Evaluation ***** {self.eval_mode} >>>> MeanIoU: {mIOU:.2f}\n')
    
    def configure_optimizers(self):  
        self.optimizer = optim.SGD(
            [{'params': self.model.backbone.parameters(), 'lr': self.lr},
             {'params': [param for name, param in self.model.named_parameters() if 'backbone' not in name], 'lr': self.lr * self.lr_multi}],
            lr=self.lr, momentum=0.9, weight_decay=1e-4)
        total_iters = self.total_iters // self.trainer.accumulate_grad_batches
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda iter: (1 - iter / total_iters) ** 0.9)
        return [self.optimizer]
     
    def configure_callbacks(self):
        self.logging_callback = LoggingCallback(monitor="val_mIOU",
                                            mode="max",
                                            log_file= os.path.join(self.save_path, 'output.log')
        )
        self.checkpoint_callback = ModelCheckpoint(monitor="val_mIOU",
                                            mode="max",
                                            save_last=True,
                                            dirpath=os.path.join(self.save_path, "checkpoints"),
                                            filename="{epoch:02d}-{val_mIOU:.2f}",
                                            save_top_k=5,
                                            verbose=True
                                              )
        return [self.logging_callback, self.checkpoint_callback]

    def sliding_window_eval(self, img):
        grid = self.crop_size
        b, _, h, w = img.shape
        final = torch.zeros(b, self.nclass, h, w).type_as(img)
        row = 0
        while row < h:
            col = 0
            while col < w:
                pred = self(img[:, :, row: min(h, row + grid), col: min(w, col + grid)], False)
                final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                col += int(grid * 2 / 3)
            row += int(grid * 2 / 3)
        pred = final.argmax(dim=1)
        return pred
    
    def center_crop_eval(self, img, mask):
        h, w = img.shape[-2:]
        start_h, start_w = (h - self.crop_size) // 2, (w - self.crop_size) // 2
        img = img[:, :, start_h:start_h + self.crop_size, start_w:start_w + self.crop_size]
        mask = mask[:, start_h:start_h + self.crop_size, start_w:start_w + self.crop_size]
        pred = self(img, False).argmax(dim=1)
        return pred

