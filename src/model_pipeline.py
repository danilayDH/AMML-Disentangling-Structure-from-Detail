import os
import wandb
import argparse

import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.cli import LightningCLI

from basic_vae_module import VAE
from callbacks import ReconstructionsCallback
from mri_data_module import MriDataModule
import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score


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


class DownstreamClassifier:

    def __init__(self, vae_model, checkpoint_dir: str, is_ukbb: bool = False, label_column: str =None):
        self.vae = vae_model
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.checkpoint = checkpoint_dir.split('/')[-1]
        self.is_ukbb = is_ukbb
        self.label_column = label_column if is_ukbb else 'DX'
        self.data_module = MriDataModule(
            data_dir="src/ukbb_small.csv" if is_ukbb else "src/adni_small.csv",
            batch_size=16, is_ukbb=is_ukbb, label_column=self.label_column
        )

    def prepare_data(self, data_loader, use_demographics=False):
        X = []
        y = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                if isinstance(labels, tuple):
                    print(labels)
                if not use_demographics:
                    _, _, z = self.vae.forward_to_latent(inputs)
                    for latent_repr in z:
                        X.append(latent_repr.numpy())
                else:
                    for input in inputs:
                        X.append(input.numpy())
                y.extend(labels.numpy())
        X = np.array(X)
        y = np.array(y)
        print(f"Prepared data X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def predict_using_demographics(self, use_test_set=False):
        print("UKBB is: ", self.is_ukbb)
        print("\nPredict using demographics:")
        print("\nLoad demographics:")
        train_dataloader = self.data_module.train_dataloader(no_mci=True, use_demographics=True, shuffle=False)
        test_dataloader = self.data_module.test_dataloader(no_mci=True, use_demographics=True)

        X_train, y_train = self.prepare_data(train_dataloader, use_demographics=True)
        X_test, y_test = self.prepare_data(test_dataloader, use_demographics=True)

        model = RidgeCV(alphas=(0.1, 1.0, 10.0)) if self.is_ukbb else LogisticRegressionCV(random_state=0, class_weight='balanced')
        print(model)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)


        if self.is_ukbb:
            print("\nDemographics - Train Set Results:")
            print("R2 Score on Train Set:", r2_score(y_train, y_pred_train))
            
            print("\nDemographics - Test Set Results:")
            print("R2 Score on Test Set:", r2_score(y_test, y_pred_test))
        else:
            print("\nDemographics - Train Set Results:")
            print("Balanced Accuracy on Train Set:", balanced_accuracy_score(y_train, y_pred_train))

            print("\nDemographics - Test Set Results:")
            print("Balanced Accuracy on Test Set:", balanced_accuracy_score(y_test, y_pred_test))

        if self.is_ukbb:
            np.save(self.checkpoint + "_ukbb_demographics_predict_probs.npy", y_pred_test)
        else:
            np.save(self.checkpoint + "_adni_demographics_predict_probs.npy", model.predict_proba(X_test))

    def predict_using_vae(self, use_test_set=False):
        print("\nPredict using VAE:")
        print("\nLoad latent representations:")
        train_dataloader_lat = self.data_module.train_dataloader(no_mci=True, shuffle=False)
        test_dataloader_lat = self.data_module.test_dataloader(no_mci=True)

        X_train, y_train = self.prepare_data(train_dataloader_lat, use_demographics=False)
        X_test, y_test = self.prepare_data(test_dataloader_lat, use_demographics=False)

        model = RidgeCV(alphas=(0.1, 1.0, 10.0)) if self.is_ukbb else LogisticRegressionCV(random_state=0, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        print("UKBB is: ", self.is_ukbb)

        if self.is_ukbb:
            print("\nVAE - Train Set Results:")
            print("R2 Score on Training Set:", r2_score(y_train, y_pred_train))

            print("\nVAE - Test Set Results:")
            print("R2 Score on Test Set:", r2_score(y_test, y_pred_test))
        else:
            print("\nVAE - Train Set Results:")
            print("Balanced Accuracy on Training Set:", balanced_accuracy_score(y_train, y_pred_train))

            print("\nVAE - Test Set Results:")
            print("Balanced Accuracy on Test Set:", balanced_accuracy_score(y_test, y_pred_test))

    def predict_using_both(self, use_test_set=False):
        print("\nPredict using VAE and demographics:")
        print("\nLoad demographics:")
        train_dataloader_dem = self.data_module.train_dataloader(no_mci=True, use_demographics=True, shuffle=False)
        test_dataloader_dem = self.data_module.test_dataloader(no_mci=True, use_demographics=True)
        X_train_dem, y_train = self.prepare_data(train_dataloader_dem, use_demographics=True)
        X_test_dem, y_test = self.prepare_data(test_dataloader_dem, use_demographics=True)

        print("\nLoad latent representations:")
        train_dataloader_lat = self.data_module.train_dataloader(no_mci=True, shuffle=False)
        test_dataloader_lat = self.data_module.test_dataloader(no_mci=True)
        X_train_lat, _ = self.prepare_data(train_dataloader_lat, use_demographics=False)
        X_test_lat, _ = self.prepare_data(test_dataloader_lat, use_demographics=False)

        X_train = np.concatenate((X_train_dem, X_train_lat), axis=1)
        X_test = np.concatenate((X_test_dem, X_test_lat), axis=1)

        model = RidgeCV(alphas=(0.1, 1.0, 10.0)) if self.is_ukbb else LogisticRegressionCV(random_state=0, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        if self.is_ukbb:
            print("\nVAE & demographics- Train Set Results:")
            print("R2 Score on Training Set:", r2_score(y_train, y_pred_train))

            print("\nVAE & demographics - Test Set Results:")
            print("R2 Score on Test Set:", r2_score(y_test, y_pred_test))
        else:
            print("\nVAE & demographics - Train Set Results:")
            print("Balanced Accuracy on Training Set:", balanced_accuracy_score(y_train, y_pred_train))

            print("\nVAE & demographics - Test Set Results:")
            print("Balanced Accuracy on Test Set:", balanced_accuracy_score(y_test, y_pred_test))

        print("\nAnalyzing correlation between latent representations and age and sex:")

        lr_model = LinearRegression()
        r2_score_train = lr_model.fit(X_train_lat, X_train_dem).score(X_train_lat, X_train_dem)

        print("R2 score for predicting demographics from latent representations (train set):", r2_score_train)

        r2_score_test = lr_model.score(X_test_lat, X_test_dem)
        print("R2 score for predicting demographics from latent representatios (test set):", r2_score_test)

def cli_main():
    torch.set_float32_matmul_precision('medium')

    logger = pytorch_lightning.loggers.WandbLogger(project='Brain Disentanglement Dementia', save_dir='logs',
                                                   log_model=False)
  

    cli = LightningCLI(save_config_callback=None, run=False,
                       trainer_defaults={'logger': logger, 'accelerator': 'auto', 'devices': 1,
                                         'deterministic': True, 'log_every_n_steps': 50,
                                         'callbacks': [TQDMProgressBar(refresh_rate=1)]},
                       datamodule_class=MriDataModule,
                       model_class=VAE)
    
    # Seed everything using the seed provided by --seed_everything
    if cli.config['seed_everything']:
        seed = cli.config['seed_everything']
        from lightning_fabric.utilities.seed import seed_everything
        seed_everything(seed, workers=True)
         
        
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
     
        trainer.fit(cli.model, datamodule=data_module)

        # Save the trained model
        trained_vae_path = os.path.join(checkpoint_dir, f"fold_{fold_idx}_trained_vae.pth")
        torch.save(cli.model.state_dict(), trained_vae_path)

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

    # Load the trained VAE model for downstream tasks
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
    trained_vae = VAE.load_from_checkpoint(last_checkpoint_path)
    trained_vae.eval()  # Set the model to evaluation mode

    # Downstream tasks on both ADNI and UKBB data
    print("\nPerforming downstream tasks on ADNI data:")
    classifier_adni = DownstreamClassifier(trained_vae, checkpoint_dir, is_ukbb=False)
    classifier_adni.predict_using_demographics()
    classifier_adni.predict_using_vae()
    classifier_adni.predict_using_both()

    print("\nPerforming downstream tasks on UKBB data:")
    classifier_ukbb = DownstreamClassifier(trained_vae, checkpoint_dir, is_ukbb=True, 
                                           label_column=cli.config['data']['label_column'])
    classifier_ukbb.predict_using_demographics()
    classifier_ukbb.predict_using_vae()
    classifier_ukbb.predict_using_both()

if __name__ == '__main__':
    cli_main()
