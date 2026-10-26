import random

from PIL import Image

from img2text.synth.augment import AugmentConfig, augment


def _img():
    return Image.new("RGB", (200, 120), "white")


def test_augment_keeps_size_and_records_params():
    res = augment(_img(), random.Random(1), AugmentConfig(p_blur=1, p_jpeg=1, p_noise=1,
                                                          p_rotate=1, p_crop=1, p_color=1,
                                                          p_resample=1))
    assert res.image.size == (200, 120)
    assert res.image.mode == "RGB"
    assert {"blur_radius", "jpeg_quality", "noise_sigma", "rotate_deg", "crop_margin_frac",
            "resample_factor", "brightness", "contrast"} <= set(res.params)


def test_no_augmentation_is_identity_in_params():
    res = augment(_img(), random.Random(1), AugmentConfig(0, 0, 0, 0, 0, 0, 0))
    assert res.params == {}
    assert res.image.tobytes() == _img().tobytes()


def test_input_image_not_modified():
    src = _img()
    augment(src, random.Random(2), AugmentConfig(p_blur=1, p_rotate=1))
    assert src.tobytes() == _img().tobytes()
