import pandas as pd
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from datasets import MriDataset


class MriDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "src/adni.csv", batch_size: int = 32, fold:int = 0, num_folds: int = 1, test_ratio: float = 0.20, 
                 val_ratio : float = 0.15, is_ukbb: bool = False, label_column: str = None):
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
        self.val_ratio = val_ratio
        self.is_ukbb = is_ukbb
        self.label_column = label_column or ('DX' if not is_ukbb else 'BMI') # default DX for ADNI and BMI for UKBB
        self.save_hyperparameters()
        self.data = pd.read_csv(data_dir)
    
        # Adjust column names based on the dataset
        self.id_column = 'eid' if self.is_ukbb else 'PTID'
        
        self.subjects = self.data[self.id_column].unique()
        self.train_subjects, self.val_subjects, self.test_subjects = self.split_subjectwise()

        # Print out the lengths of train_subjects, val_subjects, and test_subjects
         
        print(f"Fold {self.fold + 1}/{self.num_folds}")
        print("Number of subjects in train set:", len(self.train_subjects))
        print("Number of subjects in validation set:", len(self.val_subjects))
        print("Number of subjects in test set:", len(self.test_subjects))

    
    def setup(self, stage: str):
        pass

    def _remove_nan_and_mci(self, data, no_mci=False):
        nan_count = data[self.label_column].isna().sum()
        print(f"Number of NaN values in {self.label_column}: {nan_count}. Removing those before creating the dataset.")
        data = data.dropna(subset=[self.label_column])

        if not self.is_ukbb and no_mci:
            data = data[data[self.label_column] != 'MCI']
        
        return data.reset_index(drop=True)

    def train_dataloader(self, no_mci: bool = False, use_demographics: bool = False, shuffle: bool = True):
        train_data = self.data[self.data[self.id_column].isin(self.train_subjects)]       

        train_data = self.data[self.data[self.id_column].isin(self.train_subjects)]
        train_data = self._remove_nan_and_mci(train_data, no_mci)
        train_dataset = MriDataset(train_data, axis_view="coronal", use_demographics=use_demographics, transform=None, 
                                   is_ukbb=self.is_ukbb, label_column=self.label_column)

        print("Train dataset size: " + str(len(train_data)))
        print(f"Batch size in train_dataloader: {self.batch_size}") 
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=10,
            shuffle=shuffle,
            drop_last=True 
        )

    def val_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        if self.num_folds == 1:
            return None

        val_data = self.data[self.data[self.id_column].isin(self.val_subjects)]

        val_data = self.data[self.data[self.id_column].isin(self.val_subjects)]
        val_data = self._remove_nan_and_mci(val_data, no_mci)
        val_dataset = MriDataset(val_data, axis_view="coronal", use_demographics=use_demographics, transform=None, 
                                   is_ukbb=self.is_ukbb, label_column=self.label_column)
        print("Validation dataset size: " + str(len(val_data)))
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=5
        )

    def test_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        test_data = self.data[self.data[self.id_column].isin(self.test_subjects)]
             
        test_data = self.data[self.data[self.id_column].isin(self.test_subjects)]
        test_data = self._remove_nan_and_mci(test_data, no_mci)
        test_dataset = MriDataset(test_data, axis_view="coronal", use_demographics=use_demographics, transform=None, 
                                   is_ukbb=self.is_ukbb, label_column=self.label_column)
        
        print("Test dataset size: " + str(len(test_data)))
        return DataLoader(
            test_dataset, 
            batch_size=self.batch_size
            )

    def predict_dataloader(self, no_mci: bool = False, use_demographics: bool = False):
        predict_data = self.data[self.data[self.id_column].isin(self.val_subjects)]
        
        predict_data = self.data[self.data[self.id_column].isin(self.val_subjects)]
        predict_data = self._remove_nan_and_mci(predict_data, no_mci)
        predict_dataset = MriDataset(predict_data, axis_view="coronal", use_demographics=use_demographics, transform=None, 
                                   is_ukbb=self.is_ukbb, label_column=self.label_column)
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

        if self.num_folds == 1:
            train_subjects = remaining_subjects
            val_subjects = []
        else: 
            kfold = KFold(n_splits=self.num_folds, shuffle=False)
            train_val_indices = np.where(np.isin(self.subjects, remaining_subjects))[0]
            train_subjects = []
            val_subjects = []
            for i, (train_indices, val_indices) in enumerate(kfold.split(train_val_indices)):
                if i == self.fold:
                    train_subjects = self.subjects[train_indices]
                    val_subjects = self.subjects[val_indices]
                    break
        
        return train_subjects, val_subjects, test_subjects
