import os
import wandb

import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
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

def calculate_metrics(images, reconstructions):
    mse_list = []
    snr_list = []
    for i in range(len(images)):
        mse = calculate_mse(images[i], reconstructions[i])
        snr = calculate_snr(images[i], reconstructions[i])
        mse_list.append(mse)
        snr_list.append(snr)
    return torch.stack(mse_list).mean().item(), torch.stack(snr_list).mean().item()

def cli_main():

    torch.set_float32_matmul_precision('medium')

    logger = pytorch_lightning.loggers.WandbLogger(project='Brain Disentanglement Dementia', save_dir='logs',
                                                   log_model=False)

    from lightning_fabric.utilities.seed import seed_everything
    seed_everything(176988783, workers=True)

    cli = LightningCLI(save_config_callback=None, run=False,
                       trainer_defaults={'logger': logger, 'accelerator': 'auto', 'devices': 1,
                                         'deterministic': True, 'log_every_n_steps': 50,
                                         'callbacks': [TQDMProgressBar(refresh_rate=1)]},
                       datamodule_class=MriDataModule,
                       model_class=VAE)
    
    # Print CUDA device information
    local_rank = os.getenv('LOCAL_RANK', 0)
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '[0]')
    print(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    
    data_module = cli.datamodule
    num_folds = data_module.num_folds

    # Iterate over folds
    for fold_idx in range(data_module.num_folds):
        # Print current fold number
        print(f"Processing fold {fold_idx + 1} of {data_module.num_folds}")

        # Create new data module instance for current fold
        data_module = MriDataModule(data_dir=data_module.data_dir, batch_size=data_module.batch_size,
                                    fold=fold_idx, num_folds=num_folds, test_ratio=data_module.test_ratio)
        
        # Reset data loaders
        data_module.setup(stage='fit')

        # Print dataset sizes only once per fold
        val_loader = data_module.val_dataloader()
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Batch size in val_dataloader: {data_module.batch_size}")

        cli.model = VAE(**cli.model.hparams)  # Re-instantiate model for each fold
        cli.trainer.callbacks = []  # Clear existing callbacks

        # Add ReconstructionsCallback
        reconstructions_callback = ReconstructionsCallback(dataloader=data_module.val_dataloader(), num_images=8)
        cli.trainer.callbacks.append(reconstructions_callback)

        # Add ModelCheckpoint
        checkpoint_dir = os.path.join("checkpoints", logger.experiment.id)
        checkpoint_callback = ModelCheckpoint(monitor="val/recon/loss", mode="min", save_top_k=3, save_last=True,
                                              dirpath=checkpoint_dir, auto_insert_metric_name=False,
                                              filename="fold_{fold_idx}-epoch{epoch}-{val/recon/loss:.2f}")
        cli.trainer.callbacks.append(checkpoint_callback)
        cli.trainer.callbacks.append(TQDMProgressBar(refresh_rate=1))
        
        # Fit the model
        trainer = cli.trainer.fit(cli.model, datamodule=data_module)

        # Calculate MSE and SNR
        mse_list = []
        snr_list = []

        batch_count = 0  # Initialize batch_count

        for batch in data_module.val_dataloader():
            batch_count += 1
            [stacked_image, mask_tensor], _ = batch
            
            with torch.no_grad():
                recon_batch = cli.model((stacked_image, mask_tensor))
            
            mse = calculate_mse(stacked_image, recon_batch)
            snr = calculate_snr(stacked_image, recon_batch)

            mse_list.append(mse)
            snr_list.append(snr)

        # Average MSE and SNR across the validation set
        avg_mse = torch.stack(mse_list).mean().item()
        avg_snr = torch.stack(snr_list).mean().item()

        # Log metrics to Wandb
        wandb.log({'fold': fold_idx, 'avg_mse': avg_mse, 'avg_snr': avg_snr})

        del mse_list, snr_list, avg_mse, avg_snr, recon_batch, batch, stacked_image, mask_tensor


if __name__ == '__main__':
    cli_main()
