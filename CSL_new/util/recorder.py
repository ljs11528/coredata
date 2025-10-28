from .utils import AverageMeter


class Recorder:
    def __init__(self, name, log, prog_bar=False):
        self.name = name
        self.log = log
        self.prog_bar = prog_bar
        self.total_value = AverageMeter()

    def __call__(self, value):
        self.total_value.update(value)
        self.log(f'{self.name}_total', value, on_step=True, on_epoch=False, prog_bar=self.prog_bar, logger=True, rank_zero_only=True)  

    def log_metrics(self):
        result = self.total_value.avg
        self.total_value.reset()
        self.log(f'{self.name}_avg', result, on_step=True, on_epoch=False,logger=True, rank_zero_only=True)
        return f',{self.name}: {result:.3f}'
        
        
