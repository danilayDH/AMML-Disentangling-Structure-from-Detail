import argparse
import os

import torch
from basic_vae_module import VAE
from mri_data_module import MriDataModule

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score


class DownstreamClassifier:

    def __init__(self, checkpoint: str, is_ukbb: bool = False):
        self.checkpoint_path= "checkpoints/" + checkpoint + "/last.ckpt"
        self.checkpoint = checkpoint
        self.is_ukbb = is_ukbb
        self.vae = VAE.load_from_checkpoint(self.checkpoint_path)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.data_module = MriDataModule(data_dir="src/ukbb_small.csv" if is_ukbb else "src/adni.csv", 
                                         batch_size=16, downstream_task=True, is_ukbb=is_ukbb)

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

    def predict_using_demographics(self, use_test_set: bool=False):
        train_dataloader = self.data_module.train_dataloader(no_mci=True, use_demographics=True)
        val_dataloader = self.data_module.val_dataloader(no_mci=True, use_demographics=True)
        test_dataloader = self.data_module.test_dataloader(no_mci=True, use_demographics=True)
        X_train, y_train = self.prepare_data(train_dataloader, use_demographics=True)
        X_val, y_val = self.prepare_data(val_dataloader, use_demographics=True)
        X_test, y_test = self.prepare_data(test_dataloader, use_demographics=True)
        
        if self.is_ukbb:
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        else:
            model = LogisticRegression(random_state=0)
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        if self.is_ukbb:
            print("Demographics - Train Set Results:")
            print("R2 Score on Train Set:", r2_score(y_train, y_pred_train))
            
            print("\nDemographics - Validation Set Results:")
            print("R2 Score on Validation Set:", r2_score(y_val, y_pred_val))
            
            print("\nDemographics - Test Set Results:")
            print("R2 Score on Test Set:", r2_score(y_test, y_pred_test))
        else:
            print("Demographics - Train Set Results:")
            print("Balanced Accuracy on Train Set:", balanced_accuracy_score(y_train, y_pred_train))
            
            print("\nDemographics - Validation Set Results:")
            print("Balanced Accuracy on Validation Set:", balanced_accuracy_score(y_val, y_pred_val))
            
            print("\nDemographics - Test Set Results:")
            print("Balanced Accuracy on Test Set:", balanced_accuracy_score(y_test, y_pred_test))
        
        if self.is_ukbb:
            np.save(self.checkpoint + "_ukbb_demographics_predict_probs.npy", y_pred_val)
        else:
            np.save(self.checkpoint + "_adni_demographics_predict_probs.npy", model.predict_proba(X_val))
        
       
    def predict_using_vae(self, use_test_set: bool=False):
        if self.is_ukbb:
            pred_file = self.checkpoint + "_ukbb_demographics_predict_probs.npy"
        else:
            pred_file = self.checkpoint + "_adni_demographics_predict_probs.npy"
        
        
        if os.path.exists(pred_file):
            demographics_pred_prob = np.load(self.checkpoint + "_demographics_predict_probs.npy")
            print("Predictions on Validation Set based on demographics:", demographics_pred_prob)
        else:
            print("Demographics predictions not found.")
            return
        
        print("Predict using VAE and demographics:")
        print("\nLoad demographics:")
        train_dataloader_dem = self.data_module.train_dataloader(no_mci=True, use_demographics=True)
        val_dataloader_dem = self.data_module.val_dataloader(no_mci=True, use_demographics=True)
        test_dataloader_dem = self.data_module.test_dataloader(no_mci=True, use_demographics=True)

        X_train_dem, y_train = self.prepare_data(train_dataloader_dem, use_demographics=True)
        X_val_dem, y_val = self.prepare_data(val_dataloader_dem, use_demographics=True)
        X_test_dem, y_test = self.prepare_data(test_dataloader_dem, use_demographics=True)

        print("\nLoad latent representations:")

        train_dataloader_lat = self.data_module.train_dataloader(no_mci=True)
        val_dataloader_lat = self.data_module.val_dataloader(no_mci=True)
        test_dataloader_lat = self.data_module.test_dataloader(no_mci=True)
        X_train_lat, _ = self.prepare_data(train_dataloader_lat, use_demographics=False)
        X_val_lat, _ = self.prepare_data(val_dataloader_lat, use_demographics=False)
        X_test_lat, _ = self.prepare_data(test_dataloader_lat, use_demographics=False)

        # Concatenate demographic and latent features
        X_train = np.concatenate((X_train_dem, X_train_lat), axis=1)
        X_val = np.concatenate((X_val_dem, X_val_lat), axis=1)
        X_test = np.concatenate((X_test_dem, X_test_lat), axis=1)
        
        if self.is_ukbb:
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        else:
            model = LogisticRegression(random_state=0)
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        if self.is_ukbb:
            print("VAE - Train Set Results:")
            print("R2 Score on Training Set:", r2_score(y_train, y_pred_train))

            print("\nVAE - Validation Set Results:")
            print("R2 Score on Validation Set:", r2_score(y_val, y_pred_val))

            print("\nVAE - Test Set Results:")
            print("R2 Score on Test Set:", r2_score(y_test, y_pred_test))
        else:
            print("VAE - Train Set Results:")
            print("Balanced Accuracy on Training Set:", balanced_accuracy_score(y_train, y_pred_train))

            print("\nVAE - Validation Set Results:")
            print("Balanced Accuracy on Validation Set:", balanced_accuracy_score(y_val, y_pred_val))

            print("\nVAE - Test Set Results:")
            print("Balanced Accuracy on Test Set:", balanced_accuracy_score(y_test, y_pred_test))
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Dementia with help of VAE")
    parser.add_argument("checkpoint", type=str, help="Checkpoint id")
    parser.add_argument("--is_ukbb", action="store_true", help="Use UKBB dataset")
    args = parser.parse_args()
    from lightning_fabric.utilities.seed import seed_everything
    seed_everything(176988783, workers=True)
    downstream_task = DownstreamClassifier(args.checkpoint, is_ukbb=args.is_ukbb)
    downstream_task.predict_using_demographics()
    downstream_task.predict_using_vae()