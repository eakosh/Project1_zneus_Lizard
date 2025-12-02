import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torchvision.utils import make_grid
import torch
import wandb
import glob
from torchvision import transforms as T


COLORS = {
    0: (0, 0, 0),          # background
    1: (255, 0, 0),        # neutrophil
    2: (0, 255, 0),        # epithelial
    3: (0, 0, 255),        # lymphocyte
    4: (255, 255, 0),      # plasma
    5: (255, 0, 255),      # eosinophil
    6: (0, 255, 255),      # connective tissue
}


def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, rgb in COLORS.items():
        color[mask == cls] = rgb

    return color


class SegmentationVisualizer(pl.Callback):
    def __init__(self, 
                 val_img_dir="patches/val/img",
                 val_mask_dir="patches/val/mask",
                 num_samples=3,
                 every_n_epochs=10):

        super().__init__()
        self.val_img_dir = val_img_dir
        self.val_mask_dir = val_mask_dir
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

        self.img_paths = sorted(glob.glob(f"{val_img_dir}/*.png"))

        self.img_paths = self.img_paths[:num_samples]

        self.to_tensor = T.ToTensor()


    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.every_n_epochs != 0:
            return

        if not hasattr(trainer.logger, "experiment"):
            return

        logs = {}

        vis_list = []

        for path in self.img_paths:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)

            mask_path = path.replace("/img/", "/mask/")
            true_mask = np.array(Image.open(mask_path))

            img_tensor = self.to_tensor(img)
            img_tensor = self.normalize(img_tensor)  
            x = img_tensor.unsqueeze(0).to(pl_module.device)

            logits = pl_module(x)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

            true_color = colorize_mask(true_mask)
            pred_color = colorize_mask(pred_mask)

            combined = np.concatenate([img_np, true_color, pred_color], axis=1)
            vis_list.append(wandb.Image(combined, caption=f"{path}"))

        logs["val_examples"] = vis_list
        trainer.logger.experiment.log(logs)
