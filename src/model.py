import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from losses import ComboLoss


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def per_class_iou(logits, target, num_classes, eps=1e-6):
    preds = torch.argmax(logits, dim=1)

    ious = {}
    for cls in range(num_classes):
        pred_c = (preds == cls)
        target_c = (target == cls)

        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        if union == 0:
            iou = float("nan")
        else:
            iou = inter / (union + eps)

        ious[cls] = iou

    return ious


def mean_iou(cls_ious):
    vals = [v for v in cls_ious.values() if not (v != v)]   
    if len(vals) == 0:
        return 0.0
    return sum(vals) / len(vals)


def pixel_accuracy(logits, target):
    preds = torch.argmax(logits, dim=1)
    return (preds == target).float().mean()


class UNetSegmentation(pl.LightningModule):
    def __init__(self,
                class_weights,
                in_channels=3, 
                num_classes=7, 
                learning_rate=1e-3, 
                ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Decoder
        self.up1 = Up(512 + 512, 256)
        self.up2 = Up(256 + 256, 128)
        self.up3 = Up(128 + 128, 64)
        self.up4 = Up(64 + 64, 64)

        self.outc = OutConv(64, num_classes)

        self.loss_fn = ComboLoss(
            gamma=2.0,
            ce_weight=0.3,
            focal_weight=0.5,
            dice_weight=0.2
        )


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def shared_step(self, batch, stage: str):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        cls_ious = per_class_iou(logits, masks, self.num_classes)
        miou = mean_iou(cls_ious)
        acc = pixel_accuracy(logits, masks)

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/miou", miou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for cls, iou in cls_ious.items():
            self.log(f"{stage}/iou_class_{cls}", iou, on_epoch=True, prog_bar=False)

        return loss
    
    def compute_per_class_iou(self, logits, masks):
        return per_class_iou(logits, masks, self.num_classes)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
