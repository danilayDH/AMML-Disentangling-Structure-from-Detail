import os

import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from basic_vae_module import VAE
from callbacks import ReconstructionsCallback
from mri_data_module import MriDataModule
import torch


def cli_main():

    torch.set_float32_matmul_precision('medium')

    logger = pytorch_lightning.loggers.WandbLogger(project='Brain Disentanglement Dementia', save_dir='logs',
                                                   log_model=False)

    from lightning_fabric.utilities.seed import seed_everything
    seed_everything(176988783, workers=True)

    cli = LightningCLI(save_config_callback=None, run=False,
                       trainer_defaults={'logger': logger, 'accelerator': 'auto', 'devices': 1,
                                         'deterministic': True, 'log_every_n_steps': 50, },
                       datamodule_class=MriDataModule,
                       model_class=VAE)

    data_module = cli.datamodule
    reconstructions_callback = ReconstructionsCallback(dataloader=data_module.val_dataloader(), num_images=8)
    checkpoint_dir = os.path.join("checkpoints", logger.experiment.id)
    checkpoint_callback = ModelCheckpoint(monitor="val/recon/loss", mode="min", save_top_k=3, save_last=True,
                                          dirpath=checkpoint_dir, auto_insert_metric_name=False,
                                          filename="epoch{epoch}-{val/recon/loss:.2f}")
    cli.trainer.callbacks.append(reconstructions_callback)
    cli.trainer.callbacks.append(checkpoint_callback)

    seed = cli.config['seed_everything']
    if 'seed_everything' not in logger.experiment.config or logger.experiment.config['seed_everything'] is None:
        logger.experiment.config['seed_everything'] = seed

    cli.trainer.fit(cli.model, datamodule=data_module)


if __name__ == '__main__':
    cli_main()
