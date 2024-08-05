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
         
        
    data_module = cli.datamodule
    num_folds = data_module.num_folds
    is_ukbb = data_module.is_ukbb

    all_fold_metrics = []

    # Iterate over folds
    for fold_idx in range(data_module.num_folds):
        # Print current fold number
        print(f"Processing fold {fold_idx + 1} of {data_module.num_folds}")

        # Create new data module instance for current fold
        data_module = MriDataModule(data_dir=data_module.data_dir, batch_size=data_module.batch_size,
                                    fold=fold_idx, num_folds=num_folds, test_ratio=data_module.test_ratio, is_ukbb=is_ukbb)
        
        # Reset data loaders
        data_module.setup(stage='fit')

        # Print dataset sizes only once per fold
        val_loader = data_module.val_dataloader()
        if val_loader is not None:
            print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Batch size is: {data_module.batch_size}")

        cli.model = VAE(**cli.model.hparams)  # Re-instantiate model for each fold and pass hyperparameters as keyword arguments
        cli.trainer.callbacks = []  # Clear existing callbacks

        # Determine which dataloader to use for reconstructions
        recon_dataloader = data_module.val_dataloader() if data_module.num_folds > 1 else data_module.train_dataloader()
       
        # Add ReconstructionsCallback
        reconstructions_callback = ReconstructionsCallback(
            dataloader=recon_dataloader, 
            num_images=8, 
            fold_idx=fold_idx)
        cli.trainer.callbacks.append(reconstructions_callback)

        # Add ModelCheckpoint
        checkpoint_dir = os.path.join("checkpoints", logger.experiment.id)
        
        #checkpoint_callback = ModelCheckpoint(monitor="val/recon/loss", mode="min", save_top_k=3, save_last=True,
         #                                     dirpath=checkpoint_dir, auto_insert_metric_name=False,
          #                                    filename="fold_{fold_idx}-epoch{epoch}-{val/recon/loss:.2f}")
        #cli.trainer.callbacks.append(checkpoint_callback)
        
        checkpoint_callback_last = ModelCheckpoint(
            save_last=True, 
            dirpath=checkpoint_dir,
            filename=f"fold_{fold_idx}-last"
        )
        cli.trainer.callbacks.append(checkpoint_callback_last)
        
        cli.trainer.callbacks.append(TQDMProgressBar(refresh_rate=1))

        # Create new Trainer instance
        trainer = pytorch_lightning.Trainer(max_epochs=cli.trainer.max_epochs, logger=logger, accelerator='auto', devices=1,
                             deterministic=True, log_every_n_steps=50, callbacks=cli.trainer.callbacks)
     
        # Fit the model
        trainer = trainer.fit(cli.model, datamodule=data_module)
        
        if val_loader is not None:
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

            print("MSE:", mse)
            print("MSE List:", mse_list)
            print("Avg MSE:", avg_mse)
            print("SNR:", snr)
            print("SNR List:", snr_list)
            print("Avg SNR:", avg_snr)

            # Log metrics to Wandb
            wandb.log({'fold': fold_idx, 'avg_mse': avg_mse, 'avg_snr': avg_snr})

            all_fold_metrics.append({'mse': avg_mse, 'snr': avg_snr})

            del recon_batch, batch, stacked_image, mask_tensor
    
    if all_fold_metrics:
        # Average metrics across all folds
        final_avg_mse = sum(f['mse'] for f in all_fold_metrics) / num_folds
        final_avg_snr = sum(f['snr'] for f in all_fold_metrics) / num_folds
        

        print(f"Final Average MSE: {final_avg_mse}, Final Average SNR: {final_avg_snr}")
        wandb.log({'final_avg_mse': final_avg_mse, 'final_avg_snr': final_avg_snr})


if __name__ == '__main__':
    cli_main()
