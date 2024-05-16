import os
import wandb

import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from basic_vae_module import VAE
from callbacks import ReconstructionsCallback
from mri_data_module import MriDataModule
import torch

def calculate_mse(image1, image2):
    return ((image1 - image2) ** 2).mean()

def calculate_snr(image1, image2):
    signal = torch.sum(image1 ** 2)
    noise = torch.sum((image1 - image2) ** 2)
    return 10 * torch.log10(signal / noise)

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
    #reconstructions_callback = ReconstructionsCallback(dataloader=data_module.val_dataloader(), num_images=8)
    #checkpoint_dir = os.path.join("checkpoints", logger.experiment.id)
    #checkpoint_callback = ModelCheckpoint(monitor="val/recon/loss", mode="min", save_top_k=3, save_last=True,
     #                                     dirpath=checkpoint_dir, auto_insert_metric_name=False,
      #                                    filename="epoch{epoch}-{val/recon/loss:.2f}")
    #cli.trainer.callbacks.append(reconstructions_callback)
    #cli.trainer.callbacks.append(checkpoint_callback)

    #seed = cli.config['seed_everything']
    #if 'seed_everything' not in logger.experiment.config or logger.experiment.config['seed_everything'] is None:
    #    logger.experiment.config['seed_everything'] = seed

    #cli.trainer.fit(cli.model, datamodule=data_module)

    # Iterate over folds
    for fold_idx in range(data_module.num_folds):
        # Set fold-specific data
        data_module.fold = fold_idx

        # Reset data loaders
        data_module.setup(stage='fit')

        # Add ReconstructionsCallback
        reconstructions_callback = ReconstructionsCallback(dataloader=data_module.val_dataloader(), num_images=8)
        cli.trainer.callbacks.append(reconstructions_callback)

        # Add ModelCheckpoint
        checkpoint_dir = os.path.join("checkpoints", logger.experiment.id)
        checkpoint_callback = ModelCheckpoint(monitor="val/recon/loss", mode="min", save_top_k=3, save_last=True,
                                              dirpath=checkpoint_dir, auto_insert_metric_name=False,
                                              filename="fold_{fold_idx}-epoch{epoch}-{val/recon/loss:.2f}")
        cli.trainer.callbacks.append(checkpoint_callback)

        # Fit the model
        trainer = cli.trainer.fit(cli.model, datamodule=data_module)

        # Calculate MSE and SNR
        mse_list = []
        snr_list = []

        for batch in data_module.val_dataloader():
            [stacked_image, mask_tensor], _ = batch
            #stacked_image = stacked_image.unsqueeze(0)
            print("Main: Number of dimensions of stacked_image:", stacked_image.dim())
            print("Main: Shape of stacked_image", stacked_image.shape)
            print("Main: Number of dimensions of mask_tensor:", mask_tensor.dim())
            with torch.no_grad():
                recon_batch, _ = cli.model(stacked_image)
            
            mse = calculate_mse(stacked_image, recon_batch)
            snr = calculate_snr(stacked_image, recon_batch)

            mse_list.append(mse)
            snr_list.append(snr)

        # Average MSE and SNR across the validation set
        avg_mse = torch.stack(mse_list).mean().item()
        avg_snr = torch.stack(snr_list).mean().item()

        # Log metrics to Wandb
        wandb.log({'fold': fold_idx, 'avg_mse': avg_mse, 'avg_snr': avg_snr})


if __name__ == '__main__':
    cli_main()
