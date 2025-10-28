import pytorch_lightning as pl
import logging
from logging import handlers
from .utils import count_params

class LoggingCallback(pl.Callback):
    def __init__(self, log_file='output.log', monitor='val_mIOU', mode='max'):
        super().__init__()
        self.log_file = log_file
        self.monitor = monitor
        self.mode = mode
        self.logger = None
        if mode == 'max':
            self.best = float(0)
            self.monitor_op = lambda current, best: current > best
        elif mode == 'min':
            self.best = float(1)
            self.monitor_op = lambda current, best: current < best
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def setup_logger(self, trainer):
        if trainer.global_rank == 0:
            
            self.logger = logging.getLogger(f'logger_{self.monitor}')
            self.logger.setLevel(logging.INFO)

            if not self.logger.handlers:
                file_handler = handlers.RotatingFileHandler(
                    self.log_file, 
                    encoding='utf-8', 
                    maxBytes=10**6, 
                    backupCount=5
                )
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('[%(levelname)s] %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def on_train_start(self, trainer, pl_module):
        self.setup_logger(trainer)
        if self.logger:
            self.logger.info(
            'Total params: {:.1f}M\n'.format(count_params(pl_module))
            )


    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.logger:
            metrics = trainer.callback_metrics
            current = metrics.get(self.monitor)

            if current is not None:
                if self.monitor_op(current, self.best):
                    self.best = current

            current_epoch = trainer.current_epoch
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            self.logger.info(
                f'===========> Epoch: {current_epoch}, LR: {current_lr:.5f}, Previous best {self.monitor}: {self.best:.2f}'
            )
    