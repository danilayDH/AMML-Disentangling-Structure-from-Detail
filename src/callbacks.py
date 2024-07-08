import wandb
from pytorch_lightning import Callback
import torch


def log_images_to_wandb(images, reconstructions, epoch, fold_idx):
    # The original images are grayscale,so we only take the first channel
    images_np = images[:, 0:1, :, :].detach().cpu().numpy()
    reconstructions_np = reconstructions[:, 0:1, :, :].detach().cpu().numpy()

    #wandb.log({"original_images": [wandb.Image(image, caption=f"Original Images - Epoch {epoch}", mode="L") for
    #                               image in images_np]})
    #wandb.log({"reconstructions": [wandb.Image(reconstruction, caption=f"Reconstructions - Epoch {epoch}", mode="L")
    #                               for reconstruction in reconstructions_np]})

    wandb.log({f"fold_{fold_idx}_original_images": [wandb.Image(image, caption=f"Fold {fold_idx} - Original Images - Epoch {epoch}", mode="L") 
                                                    for image in images_np]})
    wandb.log({f"fold_{fold_idx}_reconstructions": [wandb.Image(reconstruction, caption=f"Fold {fold_idx} - Reconstructions - Epoch {epoch}", mode="L") 
                                                    for reconstruction in reconstructions_np]})
    # wandb.log(..., step=epoch + (fold_idx * max_epochs))


class ReconstructionsCallback(Callback):
    def __init__(self, dataloader, num_images=3, fold_idx=0):
        super().__init__()
        self.dataloader = dataloader
        self.num_images = num_images
        self.fold_idx = fold_idx

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_reconstructions(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_reconstructions(trainer, pl_module)

    def log_reconstructions(self, trainer, pl_module):
        if self.dataloader is None:
            return
        
        x, _ = next(iter(self.dataloader))
        images = x[0]
        masks = x[1]
        images = images[0:self.num_images]
        masks = masks[0:self.num_images]
        images = images.to(pl_module.device)
        masks = masks.to(pl_module.device)

        with torch.no_grad():  # Add this line to disable gradient computation
            reconstructions = pl_module.forward([images, masks])
       
        log_images_to_wandb(images, reconstructions, trainer.current_epoch , self.fold_idx)
