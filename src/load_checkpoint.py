import torch

from basic_vae_module import VAE
from mri_data_module import MriDataModule
from torch.utils import data as torch_data

checkpoint_path = "checkpoints/gzgc6yii/last.ckpt"

datamodule = MriDataModule(data_dir="src/adni.csv", batch_size=16)

vae = VAE.load_from_checkpoint(checkpoint_path)
vae.eval()
print("Model loaded from checkpoint.")

dataloader = torch_data.DataLoader(datamodule.val_dataset, batch_size=1, shuffle=True, num_workers=0)

for i, batch in enumerate(dataloader):
    x, y = batch
    print("Analysing image: " + str(i))
    with torch.no_grad():
        _, _, z = vae.forward_to_latent(x)
        print("z: " + str(z))
    if i == 20:
        break

