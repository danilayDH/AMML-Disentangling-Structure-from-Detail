import warnings
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch
import torch.utils.data as torch_data

IMAGE_SIZE = 176

class MriDataset(torch_data.Dataset):

    def __init__(self, data, axis_view="coronal", use_demographics: bool = False, transform=None):
        self.data = data
        self.use_demographics = use_demographics

        self.transform = transform

        if axis_view not in ["axial", "sagittal", "coronal"]:
            raise ValueError("axis_view must be one of 'axial', 'sagittal', or 'coronal'.")
        self.axis_view = axis_view

        self.labels_dict = {0: 'background',
                            2: 'left cerebral white matter',
                            3: 'left cerebral cortex',
                            4: 'left lateral ventricle',
                            5: 'left inferior lateral ventricle',
                            7: 'left cerebellum white matter',
                            8: 'left cerebellum cortex',
                            10: 'left thalamus',
                            11: 'left caudate',
                            12: 'left putamen',
                            13: 'left pallidum',
                            14: '3rd ventricle',
                            15: '4th ventricle',
                            16: 'brain-stem',
                            17: 'left hippocampus',
                            18: 'left amygdala',
                            26: 'left accumbens area',
                            28: 'left ventral DC',
                            41: 'right cerebral white matter',
                            42: 'right cerebral cortex',
                            43: 'right lateral ventricle',
                            44: 'right inferior lateral ventricle',
                            46: 'right cerebellum white matter',
                            47: 'right cerebellum cortex',
                            49: 'right thalamus',
                            50: 'right caudate',
                            51: 'right putamen',
                            52: 'right pallidum',
                            53: 'right hippocampus',
                            54: 'right amygdala',
                            58: 'right accumbens area',
                            60: 'right ventral DC'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dx = str(row['DX'])

        label = 0
        if dx == 'CN':
            label = 0
        elif dx == "Dementia":
            label = 1
        else:
            pass
            # warnings.warn("DX must be either CN or Dementia, not " + dx)

        if self.use_demographics:
            sex = torch.tensor(1.0 if row['Sex'] == 'M' else 0.0, dtype=torch.float32)  # Convert 'M' to 1.0 (male) and 'F' to 0.0 (female)
            age = torch.tensor(row['Age'], dtype=torch.float32)  # Assuming 'age' is an integer column
            return torch.tensor([sex, age]), label

        path_scan = Path(row['filepath'])

        # Extract the filename and change it
        original_filename = path_scan.name
        new_filename = original_filename.replace("T1toMNIlin.nii.gz", "T1toMNIlin_synthseg.nii.gz")

        # Construct the new path with the modified filename
        path_mask = path_scan.parent / "synthseg_7_3_2" / new_filename

        if not path_scan.exists() or not path_mask.exists():
            warnings.warn(
                f"File {path_scan} or {path_mask} does not exist. Make sure you run this code on the cluster. Returning a "
                f"black image instead.", UserWarning)
            image = torch.ones((1, IMAGE_SIZE, IMAGE_SIZE))
            mask = torch.zeros((32, IMAGE_SIZE, IMAGE_SIZE))
            result = [image, mask]
            return result, label

        scan = nib.load(path_scan).get_fdata()
        scan_mask = nib.load(path_mask).get_fdata()
        if self.axis_view == "axial":
            image = scan[:, :, scan.shape[2] // 2]
            mask = scan_mask[:, :, scan_mask.shape[2] // 2]
        elif self.axis_view == "sagittal":
            image = scan[scan.shape[0] // 2, :, :]
            mask = scan_mask[scan_mask.shape[0] // 2, :, :]
        else:
            image = scan[:, scan.shape[1] // 2]
            mask = scan_mask[:, scan_mask.shape[1] // 2]

        offset = (len(image) - IMAGE_SIZE) // 2
        shrunken_image = image[offset:offset+IMAGE_SIZE, offset:offset+IMAGE_SIZE]
        shrunken_image = torch.tensor(shrunken_image, dtype=torch.float32)

        top_percentile = torch.quantile(shrunken_image, 0.98, interpolation='lower')
        shrunken_image = torch.clamp(shrunken_image, max=top_percentile)

        mean = shrunken_image.mean()
        std = shrunken_image.std()
        normalized_image = (shrunken_image - mean) / (std + 1e-7)
        stacked_image = torch.stack([normalized_image], dim=0)

        shrunken_mask = mask[offset:offset+IMAGE_SIZE, offset:offset+IMAGE_SIZE]
        shrunken_mask = torch.tensor(shrunken_mask, dtype=torch.long)

        # for the masks, we will encode the values as categorical, using one-hot encodings
        num_classes = len(self.labels_dict)

        # we will first transform the values from 0 to 60 into 32 consecutive integers instead, since not all
        # values from the range [0, 60] are used for labels
        labels_list = list(self.labels_dict.values())
        for idx, seg_mask_label in self.labels_dict.items():
            shrunken_mask[shrunken_mask == idx] = labels_list.index(seg_mask_label)

        mask_tensor = torch.LongTensor(shrunken_mask)
        mask_tensor = torch.nn.functional.one_hot(mask_tensor, num_classes=num_classes)

        mask_tensor = mask_tensor.permute(2, 0, 1).float()

        result = [stacked_image, mask_tensor]

        # Print the spatial dimensions of the images in stacked_image
        # print("Spatial dimensions of the images in stacked_image:", shrunken_image.shape)

        return result, label
