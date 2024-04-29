import argparse

import pytorch_lightning.loggers
import torch

from torchmetrics.regression import MeanSquaredError
from pytorch_lightning import Trainer

from basic_vae_module import VAE
from mri_data_module import MriDataModule

torch.set_float32_matmul_precision('medium')

logger = pytorch_lightning.loggers.WandbLogger(project='Brain Disentanglement Dementia', save_dir='logs',
                                               log_model=False)
parser = argparse.ArgumentParser(description="Testing VAE checkpoint")
parser.add_argument("checkpoint", type=str, help="Checkpoint id")
args = parser.parse_args()

from lightning_fabric.utilities.seed import seed_everything
seed_everything(176988783, workers=True)

checkpoint_path= "checkpoint_test/" + args.checkpoint + "/last.ckpt"
checkpoint = args.checkpoint
vae = VAE.load_from_checkpoint(checkpoint_path)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

data_module = MriDataModule(data_dir="src/adni.csv", batch_size=64)
trainer = Trainer(logger=logger, accelerator='auto', gpus=1, deterministic=True, log_every_n_steps=50)

mse_metric = MeanSquaredError()

def test_model():

    trainer.test(vae, datamodule=data_module)
    
    mse_scores = []
    for batch in data_module.val_dataloader():
        inputs, targets = batch
        shrunken_images = inputs[0]
        with torch.no_grad():
            outputs = vae(inputs)
        mse_score = mse_metric(shrunken_images, targets[:, :, :160, :160])
        mse_scores.append(mse_score.item())

    num_pixels = data_module.val_dataset[0][0].numel()  # Total number of pixels in the images
    mean_mse_per_pixel = sum(mse_scores) / (len(mse_scores) * num_pixels)

    return mean_mse_per_pixel

mean_mse_per_pixel = test_model()

# Print mean MSE per pixel
print("Mean MSE per pixel:", mean_mse_per_pixel)