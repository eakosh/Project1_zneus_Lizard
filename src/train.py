import os 
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import logging
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from model import UNetSegmentation 
from patch_datamodule import PatchDataModule
from visualize import SegmentationVisualizer
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

warnings.filterwarnings('ignore')
logging.getLogger('pydantic').setLevel(logging.ERROR)


def main(args):
    pl.seed_everything(42, workers=True)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print("\nLoading PNG patches dataset...")
    print(f"  Patch root: {config.DATA_ROOT}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Num workers: {config.NUM_WORKERS}")

    datamodule = PatchDataModule(
        root_dir=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    datamodule.setup()

    print(f"  Train patches: {len(datamodule.train_ds)}")
    print(f"  Val patches:   {len(datamodule.val_ds)}")
    print(f"  Test patches:  {len(datamodule.test_ds)}")
    print("Dataset ready\n")

    model = UNetSegmentation(
        in_channels=3,
        num_classes=7,
        learning_rate=config.LEARNING_RATE,
        class_weights=config.CLASS_WEIGHTS,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename="patchseg-{epoch:02d}-{val/loss:.4f}-{val/miou:.4f}",
        monitor="val/miou",      
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/miou",
        min_delta=0.001,  
        patience=config.EARLY_STOPPING_PATIENCE,
        mode="max",
        verbose=True,
        check_on_train_epoch_end=False, 
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    loggers = []

    if config.USE_WANDB:
        wandb_logger = WandbLogger(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=config.EXPERIMENT_NAME,
            save_dir=config.LOG_DIR,
            log_model=config.WANDB_LOG_MODEL,
        )
        loggers.append(wandb_logger)

        if config.WANDB_WATCH_MODEL:
            wandb_logger.watch(model, log="all", log_freq=100)

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback, 
            lr_monitor,
            SegmentationVisualizer(
                num_samples=config.VISUALIZE_NUM_SAMPLES, 
                every_n_epochs=config.VISUALIZE_EVERY_N_EPOCHS, 
                val_img_dir=config.VAL_IMG_DIR, 
                val_mask_dir=config.VAL_MASK_DIR),
                TQDMProgressBar(refresh_rate=20)],
        logger=loggers if len(loggers) > 0 else None,
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        benchmark=True,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    print("\nTrainer configuration:")
    print(f"  Accelerator: {trainer.accelerator.__class__.__name__}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Precision: {trainer.precision}\n")

    if args.resume_from_checkpoint:
        trainer.fit(model, datamodule, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule)

    best_ckpt = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None
    trainer.test(model, datamodule, ckpt_path=best_ckpt)

    print("\nTraining complete")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    if config.USE_WANDB:
        print(f"WandB: {wandb_logger.experiment.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nuclei segmentation with PNG patches")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume training",
    )
    args = parser.parse_args()

    main(args)
