import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random

from datasets import MriDataset


class MriDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "src/adni.csv", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.data = pd.read_csv(data_dir)
    
        self.subjects = self.data['PTID'].unique()

        self.train_subjects, self.val_subjects, self.test_subjects = self.split_subjectwise()

    def setup(self, stage: str):
        pass

    def train_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        train_data = self.data[self.data['PTID'].isin(self.train_subjects)]       

        if no_mci:
            train_data = train_data[train_data['DX'] != 'MCI']
            nan_count = train_data['DX'].isna().sum()
            print(f"Number of NaN values in DX: {nan_count}. Removing those before creating the train_dataset.")
            train_data = train_data.dropna(subset=['DX'])
        
        train_data = train_data.reset_index(drop=True)

        train_dataset = MriDataset(train_data, axis_view="coronal", use_demographics=use_demographics, transform=None)

        print("Train dataset size: " + str(len(train_data)))
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=10,
            shuffle=True,
            drop_last=True 
        )

    def val_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        val_data = self.data[self.data['PTID'].isin(self.val_subjects)]

        if no_mci:
            val_data = val_data[val_data['DX'] != 'MCI']
            nan_count = val_data['DX'].isna().sum()
            print(f"Number of NaN values in DX: {nan_count}. Removing those before creating the val_dataset.")
            val_data = val_data.dropna(subset=['DX'])
        
        val_data = val_data.reset_index(drop=True)

        val_dataset = MriDataset(val_data, axis_view="coronal",use_demographics=use_demographics, transform=None)
        print("Val dataset size: " + str(len(val_data)))
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=5,
        )

    def test_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        test_data = self.data[self.data['PTID'].isin(self.test_subjects)]
             
        if no_mci:
            test_data = test_data[test_data['DX'] != 'MCI']
            nan_count = test_data['DX'].isna().sum()
            print(f"Number of NaN values in DX: {nan_count}. Removing those before creating the test_dataset.")
            test_data = test_data.dropna(subset=['DX'])
        
        test_data = test_data.reset_index(drop=True)

        test_dataset = MriDataset(test_data, axis_view="coronal", use_demographics=use_demographics, transform=None)
        print("Test dataset size: " + str(len(test_data)))
        return DataLoader(test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        predict_data = self.data[self.data['PTID'].isin(self.val_subjects)]
        
        if no_mci:
            predict_data = predict_data[predict_data['DX'] != 'MCI']
            nan_count = predict_data['DX'].isna().sum()
            print(f"Number of NaN values in DX: {nan_count}. Removing those before creating the predict_dataset.")
            predict_data = predict_data.dropna(subset=['DX'])

        predict_data = predict_data.reset_index(drop=True)
        predict_dataset = MriDataset(predict_data, axis_view="coronal", use_demographics=use_demographics, transform=None)
        return DataLoader(predict_dataset, batch_size=self.batch_size)

    def teardown(self, stage: str):
        pass

    def split_subjectwise(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, kfolds=1, selected_fold=0):
        """
        Args:
            train_ratio (float): The ratio of subjects to include in the train set.
            val_ratio (float): The ratio of subjects to include in the validation set.
            test_ratio (float): The ratio of subjects to include in the test set.
            kfolds (int): The number of folds for cross-validation.
            selected_fold (int): The selected fold for cross-validation.

        Returns:
            tuple: A tuple containing the train subjects, validation subjects, and test subjects.
        """
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio should be equal to 1.0.")


        total_subjects = len(self.subjects)
        indices = list(range(total_subjects))

        train_size = int(train_ratio * total_subjects)
        val_size = int(val_ratio * total_subjects)
        train_val_indices = indices[:train_size + val_size] 

        # shuffle the list
        random.shuffle(train_val_indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_subjects = self.subjects[train_indices]
        val_subjects = self.subjects[val_indices]
        test_subjects = self.subjects[test_indices]

        return train_subjects, val_subjects, test_subjects
