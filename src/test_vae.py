import argparse

import pytorch_lightning.loggers
import torch
from pytorch_lightning import Trainer

from torch.nn.funtional import mse_loss

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

checkpoint_path= "checkpoints/" + args.checkpoint + "/last.ckpt"
checkpoint = args.checkpoint
vae = VAE.load_from_checkpoint(checkpoint_path)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

data_module = MriDataModule(data_dir="src/adni.csv", batch_size=64)
trainer = Trainer(logger=logger, accelerator='auto', gpus=1, deterministic=True, log_every_n_steps=50)

trainer.test(vae, datamodule=data_module)