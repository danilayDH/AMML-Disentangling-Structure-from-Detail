import pandas as pd
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from datasets import MriDataset


class MriDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "src/adni_small.csv", batch_size: int = 32, fold:int = 0, num_folds: int = 5, test_ratio: float = 0.20):
        """
        Args:
            data_dir (str): The path to the CSV file containing the data.
            batch_size (int): The batch size for the data loaders.
            test_ratio (float): The ratio of subjects to include in the test set. (Default: 0.20)
            num_folds (int): The number of folds to use for cross-validation. (Default: 5)
            fold (int): The fold to use for cross-validation. (Must be within the range [0, num_folds-1])
        """
        assert fold in range(num_folds), "Fold must be within the range [0, num_folds-1]."

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fold = fold
        self.num_folds = num_folds
        self.test_ratio = test_ratio
        self.save_hyperparameters()
        self.data = pd.read_csv(data_dir)
    
        self.subjects = self.data['PTID'].unique()

    def setup(self, stage: str):
        self.train_subjects, self.val_subjects, self.test_subjects = self.split_subjectwise()

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
        print(f"Batch size in train_dataloader: {self.batch_size}") 
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
        print(f"Batch size in val_dataloader: {self.batch_size}") 
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=5,
            drop_last = True
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
        return DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            drop_last = True
            )

    def predict_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        predict_data = self.data[self.data['PTID'].isin(self.val_subjects)]
        
        if no_mci:
            predict_data = predict_data[predict_data['DX'] != 'MCI']
            nan_count = predict_data['DX'].isna().sum()
            print(f"Number of NaN values in DX: {nan_count}. Removing those before creating the predict_dataset.")
            predict_data = predict_data.dropna(subset=['DX'])

        predict_data = predict_data.reset_index(drop=True)
        predict_dataset = MriDataset(predict_data, axis_view="coronal", use_demographics=use_demographics, transform=None)
        return DataLoader(
            predict_dataset, 
            batch_size=self.batch_size,
            drop_last = True
            )

    def teardown(self, stage: str):
        pass

    def split_subjectwise(self):
        """
        Returns:
            tuple: A tuple containing the train subjects, validation subjects, and test subjects taking cross validation into account.
        """

        total_subjects = len(self.subjects)
        indices = list(range(total_subjects))

        # Determine test set size
        test_size = int(self.test_ratio * total_subjects)

        # Randomly select test_subjects
        test_indices = np.random.choice(indices, size=test_size, replace=False)
        test_subjects = self.subjects[test_indices]

        # Remove test_subjects from the list  subjects
        remaining_subjects = np.delete(self.subjects, test_indices)

        kfold = KFold(n_splits=self.num_folds, shuffle=False)

        train_val_indices = np.where(np.isin(self.subjects, remaining_subjects))[0]

        train_subjects = []
        val_subjects = []
        for i, (train_indices, val_indices) in enumerate(kfold.split(train_val_indices)):
            if i == self.fold:
                train_subjects = self.subjects[train_indices]
                val_subjects = self.subjects[val_indices]
                break
        
         # Print out the lengths of train_subjects, val_subjects, and test_subjects
         
        print(f"Fold {self.fold + 1}/{self.num_folds}")
        print("Number of subjects in train set:", len(train_subjects))
        print("Number of subjects in validation set:", len(val_subjects))
        print("Number of subjects in test set:", len(test_subjects))

        return train_subjects, val_subjects, test_subjects
