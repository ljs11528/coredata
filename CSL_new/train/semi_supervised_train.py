from util.recorder import Recorder
from.supervised_train import SupervisedModule
from torch import nn
import torch
from util.PCOS import batch_class_stats, get_max_confidence_and_residual_variance

class SemiModule(SupervisedModule):   
    def __init__(self, batch_iters, alpha, nclass, **kwargs):
        super(SemiModule, self).__init__(nclass=nclass, **kwargs)
        self.batch_iters = batch_iters
        self.avg_log_interval = batch_iters // 8
        self.alpha = alpha
        self.num_classes = nclass
        # counters to keep track of pseudo-label pixels produced and selected during training
        # these are local to the process; on distributed training we aggregate in on_train_end
        self.total_pseudo_pixels = 0
        self.total_selected_pixels = 0
        self.criterion_u = nn.CrossEntropyLoss(reduction='none')
        self.save_hyperparameters(ignore=['model'])

        self.loss_recorder = Recorder('total_loss', self.log, True)
        self.loss_x_recorder = Recorder('loss_x', self.log, False)
        self.loss_u_s_recorder = Recorder('loss_u_s', self.log, False)
        self.loss_u_m_recorder = Recorder('loss_u_m', self.log, False)
        self.loss_u_fp_recorder = Recorder('loss_u_fp', self.log, False)
        self.mask_ratio_recorder = Recorder('mask_ratio', self.log, False)
        self.recorders = [
            self.loss_recorder,
            self.loss_x_recorder,
            self.loss_u_s_recorder,
            self.loss_u_m_recorder,
            self.loss_u_fp_recorder,
            self.mask_ratio_recorder,
        ]
        
    def on_train_batch_start(self, batch, batch_idx):   
        (img_u_w_mix, _, _, ignore_mask_mix, _, _) = batch['mixed']
        with torch.no_grad():
            self.pred_u_w_mix = self(img_u_w_mix, False, False).detach()
            self.weight_u_w_mix = self.get_weight(self.pred_u_w_mix.softmax(dim=1), ignore_mask_mix, num_classes=self.num_classes)
            self.mask_u_w_mix = self.pred_u_w_mix.argmax(dim=1)


    def training_step(self, batch, batch_idx):   
        ((img_x, mask_x),
        (img_u_w, img_u_s, img_u_m, ignore_mask, cutmix_box, cutmix_box2),
        (_, img_u_s_mix, _, ignore_mask_mix, _, _)) = batch['labeled'], batch['unlabeled'], batch['mixed']

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
        preds, preds_fp = self(torch.cat((img_x, img_u_w)), True)
        pred_x, pred_u_w = preds.split([num_lb, num_ulb])
        pred_u_fp = preds_fp[num_lb:]

        pred_u_w = pred_u_w.detach()
        weight_u_w = self.get_weight(pred_u_w.softmax(dim=1), ignore_mask, num_classes=self.num_classes)
        mask_u_w = pred_u_w.argmax(dim=1)

        conf_mask = weight_u_w == 1
        # cutmix_box2 = conf_mask & (cutmix_box2 == 1)

        # img_u_m[cover_mask.unsqueeze(1).expand_as(img_u_m)] = 0 

        img_u_m[cutmix_box2.unsqueeze(1).expand(img_u_m.shape) == 1] = \
            img_u_s_mix[cutmix_box2.unsqueeze(1).expand(img_u_m.shape) == 1]

        img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
            img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

        pred_u_s, pred_u_m = self(torch.cat((img_u_s, img_u_m)), False, False).chunk(2)

        mask_u_w_cutmixed, weight_u_w_cutmixed, ignore_mask_cutmixed = \
            mask_u_w.clone(), weight_u_w.clone(), ignore_mask.clone()

        mask_u_w_cutmixed2, weight_u_w_cutmixed2, ignore_mask_cutmixed2 = \
            mask_u_w.clone(), weight_u_w.clone(), ignore_mask.clone()

        mask_u_w_cutmixed[cutmix_box == 1] = self.mask_u_w_mix[cutmix_box == 1]
        weight_u_w_cutmixed[cutmix_box == 1] = self.weight_u_w_mix[cutmix_box == 1]
        ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

        mask_u_w_cutmixed2[cutmix_box2 == 1] = self.mask_u_w_mix[cutmix_box2 == 1]
        weight_u_w_cutmixed2[cutmix_box2 == 1] = self.weight_u_w_mix[cutmix_box2 == 1]
        ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

        loss_x = self.criterion_l(pred_x, mask_x)
        loss_u_s = self.semi_loss(pred_u_s, mask_u_w_cutmixed, weight_u_w_cutmixed, ignore_mask_cutmixed)
        loss_u_m = self.semi_loss(pred_u_m, mask_u_w_cutmixed2, weight_u_w_cutmixed2, ignore_mask_cutmixed2)
        loss_u_fp = self.semi_loss(pred_u_fp, mask_u_w, weight_u_w, ignore_mask)
        
        loss = (loss_x + loss_u_s * 0.25 + loss_u_m * 0.25 + loss_u_fp * 0.5) / 2.0
        
        mask_ratio = (conf_mask & (ignore_mask != 255)).sum().item() / \
            (ignore_mask != 255).sum()
        
        self.loss_recorder(loss.item())
        self.loss_x_recorder(loss_x.item())
        self.loss_u_s_recorder(loss_u_s.item())
        self.loss_u_m_recorder(loss_u_m.item())
        self.loss_u_fp_recorder(loss_u_fp.item())
        self.mask_ratio_recorder(mask_ratio)

        # ---- accumulate pseudo-label counts for diagnostics ----
        try:
            # valid pixels in unlabeled batch
            valid_mask = (ignore_mask != 255)
            batch_pseudo = int(valid_mask.sum().item())
            # selected pseudo-label pixels where conf_mask==True and valid
            batch_selected = int((conf_mask & valid_mask).sum().item())
        except Exception:
            batch_pseudo = 0
            batch_selected = 0
        self.total_pseudo_pixels += batch_pseudo
        self.total_selected_pixels += batch_selected

        if self.trainer.is_global_zero and batch_idx % self.avg_log_interval == 0:
            msg = f'Iters: {self.global_step}'
            for recorder in self.recorders:
                msg += recorder.log_metrics()
            self.logging_callback.info(msg)
        return loss


    def on_train_end(self) -> None:
        """Aggregate counters across processes (if distributed) and print summary."""
        # Try to aggregate totals across ranks when using distributed training
        total_pseudo = self.total_pseudo_pixels
        total_selected = self.total_selected_pixels
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = torch.tensor([total_pseudo, total_selected], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
                torch.distributed.barrier()
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                total_pseudo = int(tensor[0].item())
                total_selected = int(tensor[1].item())
        except Exception:
            # if distributed failing for some reason, fall back to local counts
            pass

        if hasattr(self, 'trainer') and getattr(self.trainer, 'is_global_zero', True):
            frac = (total_selected / total_pseudo) if total_pseudo > 0 else 0.0
            msg = (f"Training pseudo-label summary: total pseudo-label pixels produced = {total_pseudo:,}, "
                   f"total selected pixels = {total_selected:,}, fraction selected = {frac:.4f} ({frac*100:.2f}%)")
            # Print to stdout and also to logging callback if available
            print(msg)
            try:
                if hasattr(self, 'logging_callback') and self.logging_callback is not None:
                    self.logging_callback.info(msg)
            except Exception:
                pass


    @torch.no_grad()
    def get_weight(self, pred, ignore, num_classes, epsilon=1e-8, alpha=2.0):
        weight_mask = torch.zeros_like(ignore, device=ignore.device)
        valid_mask = (ignore != 255)

        max_confidence, scaled_residual_variance = get_max_confidence_and_residual_variance(
            pred, valid_mask, num_classes, epsilon
        )
        means, vars = batch_class_stats(max_confidence, scaled_residual_variance, num_classes)
        conf_mean = means[:, 0].view(-1, 1, 1)  
        res_mean = means[:, 1].view(-1, 1, 1)  
        conf_var = vars[:, 0].view(-1, 1, 1)  
        res_var = vars[:, 1].view(-1, 1, 1)   

        conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon)
        res_z = (res_mean - scaled_residual_variance) / torch.sqrt(res_var + epsilon)

        weight_conf = torch.exp(- (conf_z ** 2) / alpha) 
        weight_res = torch.exp(- (res_z ** 2) / alpha)   

        weight = weight_conf * weight_res 

        confident_mask = (conf_z > 0) | (res_z > 0) 

        weight = torch.where(confident_mask, torch.ones_like(weight), weight)
        weight_mask = torch.where(valid_mask, weight, torch.zeros_like(weight))

        return weight_mask

    
    def semi_loss(self, pred, label, weight, ignore):
        valid_mask = (ignore != 255)
        loss_high_conf = self.criterion_u(pred, label)
        loss_high_conf = loss_high_conf * weight 
        loss_u = loss_high_conf.sum() / valid_mask.sum()
        return loss_u
