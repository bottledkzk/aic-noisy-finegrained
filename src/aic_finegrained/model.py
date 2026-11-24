"""One CLIP visual tower, one classification head and five-view pooling."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPModel

from .data import FiveViews, Images, grouping_transform


class VisualClassifier(nn.Module):
    def __init__(self, base_model, classes, hidden=2048):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(str(base_model), local_files_only=True)
        self.head = nn.Sequential(nn.Linear(self.clip.config.projection_dim, hidden),
                                  nn.GELU(), nn.Linear(hidden, classes))
        self.clip.requires_grad_(False)
        self.clip.vision_model.requires_grad_(True)
        self.clip.visual_projection.requires_grad_(True)

    def project(self, pixels):
        output = self.clip.get_image_features(pixel_values=pixels, interpolate_pos_encoding=True)
        if isinstance(output, torch.Tensor):
            return output
        for field in ("image_embeds", "pooler_output"):
            value = getattr(output, field, None)
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("CLIP did not return image projection features")

    def features(self, pixels):
        projection = self.project(pixels)
        return projection / projection.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, pixels, feature_noise=0.0):
        features = self.features(pixels)
        if feature_noise:
            features = features + feature_noise * torch.randn_like(features)
        return self.head(features)

    def load_visual_weights(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(f"Incompatible checkpoint parameter names: missing={missing[:4]}, extra={extra[:4]}")
        self.load_state_dict(state, strict=False)


def execution_device(choice):
    device = torch.device("cuda" if choice == "auto" and torch.cuda.is_available()
                          else "cpu" if choice == "auto" else choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access a GPU")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def precision(device, enabled=True):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def image_loader(paths, transform, batch, workers, device, *, shuffle=False):
    return DataLoader(Images(paths, transform), batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda", drop_last=False)


@torch.inference_mode()
def extract_features(model, paths, size, batch, workers, device, *, five_view=True):
    model.eval()
    vision = model.clip.config.vision_config
    transform = FiveViews(size, vision.patch_size) if five_view else grouping_transform(vision.image_size)
    features, logits = [], []
    for pixels, _ in image_loader(paths, transform, batch, workers, device):
        pixels = pixels.to(device, non_blocking=True)
        with precision(device, enabled=five_view):
            projected = model.project(pixels.flatten(0, 1) if five_view else pixels)
        projected = projected.float()
        if five_view:
            projected = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            projected = projected.reshape(len(pixels), pixels.shape[1], -1).mean(1)
        pooled = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        features.append(pooled.cpu().numpy())
        if five_view:
            with precision(device):
                logits.append(model.head(pooled).float().cpu().numpy())
    return np.concatenate(features), np.concatenate(logits) if five_view else None
