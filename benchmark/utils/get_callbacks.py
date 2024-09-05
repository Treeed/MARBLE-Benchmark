from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback, TQDMProgressBar

import pytorch_lightning as pl
import os

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, *args):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

def get_callbacks(args):
    monitor_mode = 'min' if 'loss' in args.logger.monitor else 'max'
    checkpoint_naming = '{epoch}-{step}-{valid_loss:.4f}'
    if args.logger.monitor != 'valid_loss':
        checkpoint_naming += '-{' + args.logger.monitor + ':.4f}'
    checkpoint_callback = ModelCheckpoint(
        filename=checkpoint_naming, 
        monitor=args.logger.monitor,
        mode=monitor_mode, 
        save_top_k=1)
    
    early_stop_callback = EarlyStopping(
        args.logger.monitor, 
        patience=args.scheduler.earlystop_patience, 
        mode=monitor_mode, 
        verbose=True)
    return [checkpoint_callback, early_stop_callback, TQDMProgressBar(refresh_rate=1), CheckpointEveryNSteps(10)]

