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




#  Virchow2 

class Virchow2Backbone(nn.Module):
    def __init__(self, freeze_encoder: bool = True):
        super().__init__()

        self.encoder = timm.create_model(
            "hf_hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU
        )

        self.embed_dim = 1280 
        self.patch_grid = 16   # 224 / 14 = 16 → 16x16 

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)                # B x 261 x 1280
        patch_tokens = out[:, 5:, :]         #  cls + 4 register → B x 256 x 1280

        B, N, C = patch_tokens.shape
        h = w = int(N ** 0.5)                # 16
        feat = patch_tokens.transpose(1, 2).reshape(B, C, h, w)  # B x 1280 x 16 x 16
        return feat



class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, pool_sizes=(1, 2, 3, 6), out_channels: int = 256):
        super().__init__()
        self.stages = nn.ModuleList()
        for ps in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        priors = [x]
        for stage in self.stages:
            priors.append(
                F.interpolate(
                    stage(x),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(priors, dim=1)  # B x (in + len(pool_sizes)*out) x H x W



class Virchow2UPerDecoder(nn.Module):
    def __init__(
        self,
        in_channels_virchow: int = 1280,
        ppm_pool_sizes=(1, 2, 3, 6),
        ppm_out_channels: int = 256,
        fpn_dim: int = 256,
        num_classes: int = 7,
    ):
        super().__init__()

        self.ppm = PyramidPoolingModule(
            in_channels=in_channels_virchow,
            pool_sizes=ppm_pool_sizes,
            out_channels=ppm_out_channels
        )

        self.ppm_out_channels = in_channels_virchow + len(ppm_pool_sizes) * ppm_out_channels

        self.enc_c4 = nn.Conv2d(self.ppm_out_channels, fpn_dim, kernel_size=1)       # 16x16
        self.enc_c3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)  # 8x8
        self.enc_c2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)  # 4x4
        self.enc_c1 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)  # 2x2

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

    def forward(self, virchow_feat: torch.Tensor) -> torch.Tensor:

        x = self.ppm(virchow_feat)  # B x ppm_out_channels_total x 16 x 16

        c4 = self.enc_c4(x)      # B x fpn_dim x 16 x 16
        c3 = self.enc_c3(c4)     # B x fpn_dim x 8 x 8
        c2 = self.enc_c2(c3)     # B x fpn_dim x 4 x 4
        c1 = self.enc_c1(c2)     # B x fpn_dim x 2 x 2

        h, w = c4.size(2), c4.size(3)

        p4 = c4
        p3 = F.interpolate(c3, size=(h, w), mode="bilinear", align_corners=False)
        p2 = F.interpolate(c2, size=(h, w), mode="bilinear", align_corners=False)
        p1 = F.interpolate(c1, size=(h, w), mode="bilinear", align_corners=False)

        fpn_out = torch.cat([p4, p3, p2, p1], dim=1)  # B x (fpn_dim*4) x 16 x 16
        fpn_out = self.fpn_fuse(fpn_out)             # B x fpn_dim x 16 x 16

        logits = self.classifier(fpn_out)            # B x num_classes x 16 x 16
        return logits



class Virchow2UPerNetSegmentation(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.backbone = Virchow2Backbone(freeze_encoder=freeze_encoder)

        self.decoder = Virchow2UPerDecoder(
            in_channels_virchow=self.backbone.embed_dim,
            num_classes=num_classes,
        )

        self.loss_fn = ComboLoss(
            gamma=2.0,
            ce_weight=0.3,
            focal_weight=0.5,
            dice_weight=0.2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        virchow_feat = self.backbone(x)          # B x 1280 x 16 x 16
        logits_small = self.decoder(virchow_feat)  # B x num_classes x 16 x 16

        # upsample (H x W)
        logits = F.interpolate(
            logits_small,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        return logits

    def shared_step(self, batch, stage: str):
        images, masks = batch               # masks: B x H x W, long
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

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)