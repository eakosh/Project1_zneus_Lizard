import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



def dice_coeff(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = target.float()

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_coeff(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = target.float()

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    total = (pred + target).sum(dim=(1, 2, 3))
    union = total - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


class BaselineCNN(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=1, learning_rate=1e-3):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), 
            nn.ReLU(),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

        self.loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out_conv(x)
        return x 


    def shared_step(self, batch, stage="train"):
        images, masks = batch


        logits = self(images)  

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()

        loss = self.loss_fn(logits, masks)

        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, masks)
        iou = iou_coeff(probs, masks)

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
