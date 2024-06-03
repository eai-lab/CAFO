import os
import sys
import pickle
import datetime
import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils import print_options, setup_neptune_logger


torch.set_printoptions(sci_mode=False)
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="cafo_config")
def main(cfg: DictConfig) -> None:
    print_options(cfg)
    pl.seed_everything(cfg.seed)

    logger = setup_neptune_logger(cfg)
    # Save model checkpoint on this path
    checkpoint_path = os.path.join("checkpoints/", str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    if cfg.task.task_name in ["gilon_activity"]:
        log.info(f"Setting up GILON Dataset with CV number {cfg.task.validation_cv_num}")
        from dataset.gilon_dataset import CAFODataModule
    elif cfg.task.task_name in ["microsoft_activity"]:
        log.info(f"Setting up Microsoft Dataset with CV number {cfg.task.validation_cv_num}")
        from dataset.microsoft_dataset import CAFODataModule
    elif cfg.task.task_name in ["fingergesture"]:
        log.info(f"Setting up FingerGesture Dataset with CV number {cfg.task.validation_cv_num}")
        from dataset.finger_dataset import CAFODataModule
    elif cfg.task.task_name in ["squid_game"]:
        log.info(f"Setting up Squid Game Dataset with CV number {cfg.task.validation_cv_num}")
        from dataset.squidgame_dataset import CAFODataModule
    else:
        NotImplementedError(f"Task {cfg.task.task_name} is not implemented")

    if cfg.lightning_model == "lightning_model_cls_cafo_qr":  # Classification
        from lightning_models.lightning_model_cls_cafo_qr import LitModel
    else:
        NotImplementedError(f"Lightning model {cfg.lightning_model} is not implemented")

    # Edit channel number for ablation study
    if cfg.remove_feature:
        log.info(f"Remove feature set to True")
        if cfg.remove_feature_file_name == "none":
            cfg.task.in_channels = cfg.task.in_channels - len(cfg.remove_feature_idx_lists)
            log.info(f"Removing feature {cfg.remove_feature_idx_lists} from input channel")
        else:
            # read pickle file
            with open(
                f"importance_order_files/{cfg.task.task_name}/{cfg.model.model_name}/{cfg.remove_feature_file_name}",
                "rb",
            ) as f:
                remove_feature_file = pickle.load(f)
            log.info(f"Reading feature importance file {cfg.remove_feature_file_name}")
            print(remove_feature_file)
            cfg.remove_feature_idx_lists = remove_feature_file[cfg.remove_feature_file_key]
            cfg.task.in_channels = cfg.task.in_channels - len(cfg.remove_feature_idx_lists)
            log.info(f"Removing feature {cfg.remove_feature_idx_lists} from input channel")
    else:
        if cfg.add_random_channel_idx:
            cfg.task.in_channels = cfg.task.in_channels + 1
        else:
            # do nothing
            pass
    log.info(f"Input channel number is {cfg.task.in_channels}")
    log.info(
        f"Setting up LightningModel {cfg.lightning_model} and task {cfg.task.task_name}, model {cfg.model.model_name}"
    )

    # set up pytorch lightning model
    dm = CAFODataModule(cfg)
    dm.setup(stage="fit")
    model = LitModel(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor=cfg.task.callbacks.monitor,
        save_top_k=1,
        filename=cfg.model.model_name + "_{epoch:02d}",
        mode=cfg.task.callbacks.monitor_mode,
    )
    early_stop_callback = EarlyStopping(
        monitor=cfg.task.callbacks.monitor,
        patience=cfg.task.callbacks.patience,
        verbose=True,
        mode=cfg.task.callbacks.monitor_mode,
        min_delta=cfg.task.callbacks.min_delta,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=[cfg.gpu_id],
        check_val_every_n_epoch=cfg.task.callbacks.check_val_every_n_epoch,
        max_epochs=cfg.task.callbacks.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor_callback],
        logger=logger,
        fast_dev_run=cfg.fast_dev_run,
        log_every_n_steps=1,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )
    log.info("Start training")
    trainer.fit(model, dm)

    topk_checkpoint_paths = os.listdir(checkpoint_path)
    dm.setup("test")
    trainer.test(model, dm.test_dataloader(), ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0])

    # Flush attention files saved after best checkpoint
    if cfg.lightning_model in ["lightning_model_cls_cafo_qr", "lightning_model_cls_cafo_ce"]:
        model.flush_attention_files(checkpoint_path + "/" + topk_checkpoint_paths[0])
        model.summarize_attention_npy_files()
        log.info(f"Finished training for {cfg.lightning_model} and task {cfg.task.task_name}, cv")

        if cfg.task.task_name in ["gilon_activity"]:
            from importance_visualizer.gilon_activity_visualizer import GilonActionVisualizer
            visualizer = GilonActionVisualizer(cfg)

        elif cfg.task.task_name in ["microsoft_activity"]:
            from importance_visualizer.ms_activity_visualizer import MicrosoftActivityVisualizer
            visualizer = MicrosoftActivityVisualizer(cfg)

        elif cfg.task.task_name in ["fingergesture"]:
            from importance_visualizer.fingergesture_visualizer import FingerGestureVisualizer
            visualizer = FingerGestureVisualizer(cfg)
            
        elif cfg.task.task_name in ["squid_game"]:
            from importance_visualizer.squid_game_visualizer import SquidGameVisualizer
            visualizer = SquidGameVisualizer(cfg)
            
        visualizer.plot_global_attention()
        visualizer.plot_classwise_attention()
        logger.experiment["global_attention"].upload(visualizer.global_attention_path)
        logger.experiment["classwise_attention"].upload(visualizer.classwise_attention_path)


if __name__ == "__main__":
    # Set hyrda configuration not to change the directory by default. This is needed for the output directory to work.
    sys.argv.append("hydra.job.chdir=False")
    main()
