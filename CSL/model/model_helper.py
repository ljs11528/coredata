import importlib
import torch.nn as nn

class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self.backbone = self._build_backbone(net_cfg["backbone"])
        self.decoder = self._build_decoder(net_cfg["decoder"])
        self.saved_bn_stats = []
    def _build_backbone(self, enc_cfg):
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)
    
    def save_bn_stats(self, model):
        if not model.training:
            return
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                saved_mean = module.running_mean.clone()
                saved_var = module.running_var.clone()
                self.saved_bn_stats.append((module, saved_mean, saved_var))
                
    def restore_bn_stats(self):
        for module, saved_mean, saved_var in self.saved_bn_stats:
            module.running_mean.copy_(saved_mean)
            module.running_var.copy_(saved_var)
        self.saved_bn_stats = []
        
    def set_bn_eval(self, model, track_running_stats):
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.track_running_stats = track_running_stats

    def forward(self, x, need_fp=False, track_running_stats=True):
        self.restore_bn_stats()
        if track_running_stats == False:
            self.save_bn_stats(self.backbone)
            self.save_bn_stats(self.decoder)
        h, w = x.shape[-2:]
        c1, c2, c3, c4 = self.backbone(x)
        outs = self.decoder((c1, c2, c3, c4), h, w, need_fp)

        return outs