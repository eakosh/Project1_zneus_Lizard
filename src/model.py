import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from timm.layers import SwiGLUPacked

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

        self.att = AttentionGate(
            in_g=in_channels // 2,     
            in_x=in_channels // 2,     
            inter_channels=in_channels // 4
        )

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

        skip = self.att(x, skip)

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


class AttentionGate(nn.Module):
    def __init__(self, in_g, in_x, inter_channels):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi



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




import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from timm.layers import SwiGLUPacked


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # x: low-res; skip: high-res
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Virchow2UNIPyramid(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 7,            
        lr: float = 1e-4,
        encoder_trainable: bool = False, 
        weight_decay: float = 1e-4,
        loss_fn: Optional[nn.Module] = None,    
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])

        self.encoder = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
        )

        self.encoder_embed_dim = 1280

        if not encoder_trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

        base_ch = 256
        self.proj = nn.Conv2d(self.encoder_embed_dim, base_ch, kernel_size=1)

        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)  # 16→8
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)  # 8→4
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=3, stride=2, padding=1)  # 4→2

        self.enc1 = ConvBlock(base_ch, base_ch)           # 16×16
        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)   # 8×8
        self.enc3 = ConvBlock(base_ch * 4, base_ch * 4)   # 4×4
        self.enc4 = ConvBlock(base_ch * 8, base_ch * 8)   # 2×2

        self.up3 = UpBlock(in_ch=base_ch * 8, skip_ch=base_ch * 4, out_ch=base_ch * 4)
        self.up2 = UpBlock(in_ch=base_ch * 4, skip_ch=base_ch * 2, out_ch=base_ch * 2)
        self.up1 = UpBlock(in_ch=base_ch * 2, skip_ch=base_ch, out_ch=base_ch)

        self.seg_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, num_classes, kernel_size=1),
        )

        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss() 
        else:
            self.loss_fn = loss_fn


    def _encode_virchow_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)  # B × 261 × 1280
        patch_tokens = out[:, 5:, :]  # B × N × 1280

        B, N, C = patch_tokens.shape
        H_t = W_t = int(math.sqrt(N))
        assert H_t * W_t == N, f"The num of tokens {N} is not the square, H_t*W_t != N"

        feat = patch_tokens.transpose(1, 2).reshape(B, C, H_t, W_t)
        return feat


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        feat = self._encode_virchow_tokens(x)      # B × 1280 × H_t × W_t
        feat = self.proj(feat)                     # B × 256 × H_t × W_t

        f1 = self.enc1(feat)                       # B × 256 × H_t × W_t
        f2 = self.enc2(self.down1(f1))             # B × 512 × H_t/2 × W_t/2
        f3 = self.enc3(self.down2(f2))             # B × 1024 × H_t/4 × W_t/4
        f4 = self.enc4(self.down3(f3))             # B × 2048 × H_t/8 × W_t/8

        u3 = self.up3(f4, f3)                      # B × 1024 × H_t/4 × W_t/4
        u2 = self.up2(u3, f2)                      # B × 512 × H_t/2 × W_t/2
        u1 = self.up1(u2, f1)                      # B × 256 × H_t × W_t

        logits_tok = self.seg_head(u1)             # B × num_classes × H_t × W_t

        logits = F.interpolate(
            logits_tok,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def training_step(self, batch, batch_idx):
        imgs, masks = batch  
        logits = self(imgs)
        loss = self.loss_fn(logits, masks.long())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks.long())

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
