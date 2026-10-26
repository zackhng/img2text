import random

import pytest

from img2text.synth.chart_gen import (
    BACKENDS,
    CHART_TYPES,
    generate_spec,
    random_config,
    render,
)


def test_spec_is_reproducible():
    assert generate_spec("grouped_bar", seed=42) == generate_spec("grouped_bar", seed=42)


def test_chartcsv_has_header_and_rows():
    spec = generate_spec("line", seed=1, n_categories=5, n_series=2)
    lines = spec.to_chartcsv().splitlines()
    assert lines[0].startswith("type: line")
    assert lines[6].split(",")[0] == spec.x_title
    assert len(lines) == 7 + 5


def test_requested_bar_and_series_counts_are_honoured():
    spec = generate_spec("stacked_bar", seed=5, n_categories=8, n_series=3)
    assert len(spec.categories) == 8
    assert len(spec.series) == 3


def test_waterfall_end_equals_sum_of_steps():
    spec = generate_spec("waterfall", seed=7, n_categories=8)
    vals = spec.series["Value"]
    assert len(vals) == 8
    assert vals[-1] == pytest.approx(sum(vals[:-1]), abs=0.11)


def test_random_config_is_valid_for_type():
    rng = random.Random(0)
    for ctype in CHART_TYPES:
        for _ in range(50):
            cfg = random_config(ctype, rng)
            if ctype == "waterfall":
                assert cfg["style"].legend == "none"
            if ctype == "donut":
                assert cfg["style"].legend in ("none", "right")
            if cfg["style"].orientation == "horizontal":
                assert ctype in ("grouped_bar", "stacked_bar")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ctype", CHART_TYPES)
def test_every_type_renders_with_every_backend(tmp_path, backend, ctype):
    spec = generate_spec(ctype, seed=3)
    out = render(spec, backend, tmp_path / f"{ctype}_{backend}.png")
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ctype", CHART_TYPES)
def test_random_styles_render(tmp_path, backend, ctype):
    rng = random.Random(11)
    for i in range(2):
        cfg = random_config(ctype, rng)
        spec = generate_spec(ctype, seed=100 + i, n_categories=cfg["n_categories"],
                             n_series=cfg["n_series"], style=cfg["style"])
        out = render(spec, backend, tmp_path / f"{ctype}_{backend}_{i}.png")
        assert out.stat().st_size > 1000


def test_render_dpi_changes_pixel_size(tmp_path):
    from img2text.synth.chart_gen import Style
    small = generate_spec("line", seed=4, n_categories=5, n_series=2,
                          style=Style(render_dpi=72))
    large = generate_spec("line", seed=4, n_categories=5, n_series=2,
                          style=Style(render_dpi=220))
    a = render(small, "matplotlib", tmp_path / "small.png")
    b = render(large, "matplotlib", tmp_path / "large.png")
    from PIL import Image
    assert Image.open(b).size[0] > Image.open(a).size[0] * 2
