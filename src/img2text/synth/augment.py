"""Image augmentation that simulates real document-figure degradation.

Each operation fires with its own probability, so an augmented image may have several
degradations, one, or none. The label text is never changed by augmentation: a blurred
or rotated chart still has the same data. Returns the augmented image and the settings
applied, so every record can be traced back to its augmentation.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field

from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AugmentConfig:
    p_blur: float = 0.3
    p_jpeg: float = 0.3
    p_noise: float = 0.2
    p_rotate: float = 0.25
    p_crop: float = 0.25
    p_color: float = 0.3  # brightness / contrast jitter
    p_resample: float = 0.25  # downscale then upscale, like a low-dpi scan
    max_rotate_deg: float = 2.0
    max_crop_frac: float = 0.05


@dataclass
class AugmentResult:
    image: Image.Image
    params: dict = field(default_factory=dict)


def augment(img: Image.Image, rng: random.Random, cfg: AugmentConfig = AugmentConfig()) -> AugmentResult:
    """Apply a random subset of degradations. The input image is not modified."""
    out = img.convert("RGB")
    params: dict = {}
    w, h = out.size

    if rng.random() < cfg.p_rotate:
        angle = rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
        out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
        params["rotate_deg"] = round(angle, 2)

    if rng.random() < cfg.p_crop:
        frac = cfg.max_crop_frac
        l, t = int(w * rng.uniform(0, frac)), int(h * rng.uniform(0, frac))
        r, b = w - int(w * rng.uniform(0, frac)), h - int(h * rng.uniform(0, frac))
        out = out.crop((l, t, r, b)).resize((w, h), Image.Resampling.BICUBIC)
        params["crop_margin_frac"] = frac

    if rng.random() < cfg.p_resample:
        factor = rng.uniform(0.4, 0.8)
        small = out.resize((max(1, int(w * factor)), max(1, int(h * factor))),
                           Image.Resampling.BILINEAR)
        out = small.resize((w, h), Image.Resampling.BILINEAR)
        params["resample_factor"] = round(factor, 2)

    if rng.random() < cfg.p_blur:
        radius = rng.uniform(0.4, 1.4)
        out = out.filter(ImageFilter.GaussianBlur(radius))
        params["blur_radius"] = round(radius, 2)

    if rng.random() < cfg.p_color:
        brightness = rng.uniform(0.85, 1.15)
        contrast = rng.uniform(0.8, 1.2)
        out = ImageEnhance.Brightness(out).enhance(brightness)
        out = ImageEnhance.Contrast(out).enhance(contrast)
        params["brightness"] = round(brightness, 2)
        params["contrast"] = round(contrast, 2)

    if rng.random() < cfg.p_noise:
        sigma = rng.uniform(2.0, 8.0)
        noise = Image.effect_noise((w, h), sigma).convert("L")
        gray = Image.merge("RGB", (noise, noise, noise))
        out = Image.blend(out, gray, 0.06)
        params["noise_sigma"] = round(sigma, 2)

    if rng.random() < cfg.p_jpeg:
        quality = rng.randint(35, 85)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        params["jpeg_quality"] = quality

    return AugmentResult(image=out, params=params)
