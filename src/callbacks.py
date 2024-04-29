import wandb
from pytorch_lightning import Callback


def log_images_to_wandb(images, reconstructions, epoch):
    # The original images are grayscale,so we only take the first channel
    images_np = images[:, 0:1, :, :].cpu().numpy()
    reconstructions_np = reconstructions[:, 0:1, :, :].cpu().numpy()

    wandb.log({"original_images": [wandb.Image(image, caption=f"Original Images - Epoch {epoch}", mode="L") for
                                   image in images_np]})
    wandb.log({"reconstructions": [wandb.Image(reconstruction, caption=f"Reconstructions - Epoch {epoch}", mode="L")
                                   for reconstruction in reconstructions_np]})


class ReconstructionsCallback(Callback):
    def __init__(self, dataloader, num_images=3):
        super().__init__()
        self.dataloader = dataloader
        self.num_images = num_images

    def on_validation_epoch_end(self, trainer, pl_module):
        x, _ = next(iter(self.dataloader))
        images = x[0]
        masks = x[1]
        images = images[0:self.num_images]
        masks = masks[0:self.num_images]
        images = images.to(pl_module.device)
        masks = masks.to(pl_module.device)

        reconstructions = pl_module.forward([images, masks])
        
        log_images_to_wandb(images, reconstructions, trainer.current_epoch)
