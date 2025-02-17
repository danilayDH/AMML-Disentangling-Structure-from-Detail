import urllib.parse
from argparse import ArgumentParser

import torch
from pl_bolts import _HTTPS_AWS_HUB
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F  # noqa: N812

from components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)


class VAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.

    Model is available pretrained on different datasets:

    Example::

        # not pretrained
        vae = VAE()

        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')

        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')

    """

    pretrained_urls = {
        "cifar10-resnet18": urllib.parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-cifar10/checkpoints/epoch%3D89.ckpt"),
        "stl10-resnet18": urllib.parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-stl10/checkpoints/epoch%3D89.ckpt"),
    }

    def __init__(
            self,
            input_height: int = 176,
            enc_type: str = "resnet18",
            first_conv: bool = True,
            maxpool1: bool = False,
            enc_out_dim: int = 512,
            kl_coeff: float = 0.0001,
            latent_dim: int = 128,
            lr: float = 0.0001,
            use_segmentation_masks: str = "no_mask",
            **kwargs,
    ) -> None:
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
            use_segmentation_masks: whether and where to use segmentation masks or not
        """

        super().__init__()

        if enc_type == "resnet50":
            enc_out_dim = 2048

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.use_segmentation_masks = use_segmentation_masks

        self.save_hyperparameters()

        if use_segmentation_masks not in ['no_mask', 'in_encoder', 'in_decoder', 'separate_encoder']:
            raise Exception(
                "Variable use_segmentation_masks is not valid. It should be 'no_mask', 'in_encoder', or 'in_decoder'.")

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        self.decoder_latent_dim = self.latent_dim
        if use_segmentation_masks == "separate_encoder":
            self.decoder_latent_dim = self.latent_dim*2

        self.mask_encoder = None
        if enc_type not in valid_encoders:
            if use_segmentation_masks == "separate_encoder":
                self.mask_encoder = resnet18_encoder(first_conv, maxpool1, 32)
            if use_segmentation_masks == "in_encoder":
                self.encoder = resnet18_encoder(first_conv, maxpool1, 1+32)
            else:
                self.encoder = resnet18_encoder(first_conv, maxpool1, 1)

            self.decoder = resnet18_decoder(self.decoder_latent_dim, self.input_height, first_conv, maxpool1,
                                            use_segmentation_masks)
        else:
            if use_segmentation_masks == "separate_encoder":
                self.mask_encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1, 32)
            if use_segmentation_masks == "in_encoder":
                self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1, 1+32)
            else:
                self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1, 1)

            self.decoder = valid_encoders[enc_type]["dec"](self.decoder_latent_dim, self.input_height, first_conv, maxpool1,
                                                           use_segmentation_masks)

        self.mask_fc_mu = None
        self.mask_fc_var = None
        if use_segmentation_masks == "separate_encoder":
            self.mask_fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
            self.mask_fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        p, q, z = self.forward_to_latent(x)
        masks = x[1]
        recon_batch = self.decoder([z, masks])
        return self.decoder([z, masks])

    def _run_step(self, x):
        p, q, z = self.forward_to_latent(x)
        masks = x[1]
        return z, self.decoder([z, masks]), p, q

    def forward_to_latent(self, x):
        images = x[0]
        masks = x[1]
        
        # Change from 3d to 4d
        if images.dim() == 3:
            images = images.unsqueeze(0)

        enc_in = images
        if self.use_segmentation_masks == "in_encoder":
            enc_in = torch.cat((images, masks), dim=1)

        enc_out = self.encoder(enc_in)
        mu = self.fc_mu(enc_out)
        log_var = self.fc_var(enc_out)

        if self.use_segmentation_masks == "separate_encoder":
            mask_enc_out = self.mask_encoder(masks)
            mask_mu = self.mask_fc_mu(mask_enc_out)
            mask_log_var = self.mask_fc_var(mask_enc_out)
            mu = torch.cat((mu, mask_mu), dim=1)
            log_var = torch.cat((log_var, mask_log_var), dim=1)

        p, q, z = self.sample(mu, log_var)
        return p, q, z

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)

        std = std.clone()
        # Replace invalid values
        std[torch.isinf(std)] = 1e30
        std[torch.isnan(std)] = 1e-10
        std[std == 0.0] = 1e-10

        mu[torch.isnan(mu)] = 0.0

        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x[0], reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon/loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, batch_size=len(batch[0]))
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test/{k}": v for k, v in logs.items()}, batch_size=len(batch[0]))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default="resnet18", help="resnet18/resnet50")
        parser.add_argument("--first_conv", action="store_true")
        parser.add_argument("--maxpool1", action="store_true")
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets",
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--is_ukbb", type=bool, default=False)
        parser.add_argument("--label_column", type=str, default=None, help="Column name for the label")


        parser.add_argument("--use_segmentation_masks", type=str, default="no_mask",
                            choices=["in_encoder", "in_decoder", "no_mask", "separate_encoder"])
        
        #parser.add_argument("--seed", type=int, default=176988783, help="Random seed")

        return parser
