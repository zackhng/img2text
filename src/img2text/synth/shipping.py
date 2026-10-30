"""Oil-shipping chart domain: bilingual labels, real Brent prices, realistic tanker values.

Where the chart is about crude prices, the values are real annual averages, highs and lows
from the EIA Europe Brent spot series (`data_cache/seed/eia_brent_daily.csv`). Tanker earnings,
cargo volumes and cost drivers are synthetic, sampled in realistic ranges. Ground truth is
always the drawn data, so the chartcsv label is exact either way.

Languages: `en` (all English), `zh` (all Chinese), `mixed` (English title and axes, Chinese
categories and series names). Mixed charts are tagged `languages = [en, zh]` in their meta.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

from img2text.synth.chart_gen import ChartSpec, ChartType, Style, _series_values, random_style

SEED_CSV = Path(__file__).resolve().parents[3] / "data_cache" / "seed" / "eia_brent_daily.csv"
LAST_FULL_YEAR = 2025

# term -> (English, Chinese). Units stay English in every language so the chartcsv unit field
# is machine-readable.
TERMS: dict[str, tuple[str, str]] = {
    "average": ("Average", "平均值"),
    "high": ("High", "最高值"),
    "low": ("Low", "最低值"),
    "brent": ("Brent", "布伦特原油"),
    "vlcc": ("VLCC", "VLCC超大型油轮"),
    "suezmax": ("Suezmax", "苏伊士型油轮"),
    "aframax": ("Aframax", "阿芙拉型油轮"),
    "lr2": ("LR2 product tanker", "LR2成品油轮"),
    "middle_east": ("Middle East", "中东"),
    "americas": ("Americas", "美洲"),
    "russia": ("Russia", "俄罗斯"),
    "west_africa": ("West Africa", "西非"),
    "bunker": ("Bunker fuel", "燃油"),
    "port": ("Port fees", "港口费用"),
    "canal": ("Canal transit", "运河通行费"),
    "demurrage": ("Demurrage", "滞期费"),
    "insurance": ("War-risk insurance", "战争险保费"),
    "year": ("Year", "年份"),
    "share": ("Share", "占比"),
    "class": ("Vessel class", "船型"),
    "origin": ("Crude origin", "原油来源"),
    "cost_driver": ("Cost driver", "成本项"),
    "start": ("Voyage cost", "航次成本"),
    "end": ("Total cost", "总成本"),
}

# chart title templates keyed by chart type, (English, Chinese)
TITLES: dict[str, tuple[str, str]] = {
    "line": ("Brent spot price by year ($/bbl)", "各年布伦特原油现货价格（美元/桶）"),
    "grouped_bar": ("Tanker earnings by vessel class ($000/day)", "各船型油轮期租运价（千美元/天）"),
    "stacked_bar": ("Crude imports by origin (mb/d)", "原油进口来源（百万桶/日）"),
    "combo": ("VLCC earnings and Brent price by year", "各年VLCC运价与布伦特原油价格"),
    "donut": ("Tanker fleet by vessel class (%)", "油轮船队船型占比（%）"),
    "waterfall": ("Voyage cost bridge ($/t)", "航次成本构成（美元/吨）"),
}
AXIS: dict[str, tuple[str, str]] = {
    "value_bbl": ("Price ($/bbl)", "价格（美元/桶）"),
    "value_day": ("Earnings ($000/day)", "运价（千美元/天）"),
    "value_mbd": ("Volume (mb/d)", "进口量（百万桶/日）"),
    "value_t": ("Cost ($/t)", "成本（美元/吨）"),
    "value_pct": ("Share (%)", "占比（%）"),
    "year": ("Year", "年份"),
}

VESSEL_RANGES = {  # earnings in $000/day, sampled range
    "vlcc": (30, 90), "suezmax": (20, 60), "aframax": (20, 55), "lr2": (20, 50),
}
ORIGIN_RANGES = {  # million barrels per day
    "middle_east": (5, 12), "americas": (3, 8), "russia": (1, 4), "west_africa": (1, 3),
}
DRIVER_RANGES = {  # $ per tonne, signed change
    "bunker": (-40, 60), "port": (-10, 25), "canal": (-5, 30), "demurrage": (-10, 40),
    "insurance": (-5, 35),
}


def _t(term: str, lang: str, role: str = "category") -> str:
    en, zh = TERMS[term]
    if lang == "en":
        return en
    if lang == "zh":
        return zh
    return en if role == "axis" else zh  # mixed: English for axes, Chinese for categories/series


def _title(kind: str, lang: str) -> str:
    en, zh = TITLES[kind]
    return zh if lang == "zh" else en


def _axis(key: str, lang: str) -> str:
    en, zh = AXIS[key]
    return zh if lang == "zh" else en


@lru_cache(maxsize=1)
def load_brent_annual(path: Path = SEED_CSV) -> dict[int, tuple[float, float, float]]:
    """Year -> (average, high, low) of daily Europe Brent spot, complete calendar years only."""
    import pandas as pd

    df = pd.read_excel(path, sheet_name="Data 1", header=None, skiprows=3, names=["date", "price"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna()
    out: dict[int, tuple[float, float, float]] = {}
    for year, grp in df.groupby(df["date"].dt.year):
        if year > LAST_FULL_YEAR or len(grp) < 200:  # skip partial years
            continue
        out[int(year)] = (round(grp.price.mean(), 2), round(grp.price.max(), 2), round(grp.price.min(), 2))
    return out


def generate_shipping_spec(
    chart_type: ChartType,
    seed: int,
    lang: str = "en",
    n_categories: int | None = None,
    n_series: int | None = None,
    style: Style | None = None,
    brent: dict[int, tuple[float, float, float]] | None = None,
) -> ChartSpec:
    """Build one oil-shipping chart. Years come from the real Brent series, so the ground truth
    for crude prices is real, and tanker values are synthetic within realistic ranges."""
    if lang not in ("en", "zh", "mixed"):
        raise ValueError(f"unknown language {lang!r}")
    rng = random.Random(seed)
    if style is None:
        style = random_style(chart_type, rng)
    brent = brent or load_brent_annual()
    years_avail = sorted(brent)

    if chart_type in ("line", "grouped_bar", "stacked_bar", "combo"):
        n = n_categories or rng.choice([5, 8, 12])
        years = years_avail[-n:]
        cats = [str(y) for y in years]

    if chart_type == "line":
        names = ["average", "high", "low"]
        k = min(n_series or rng.randint(1, 3), len(names))
        pick = rng.sample(names, k)
        series = {
            _t(p, lang): [brent[y][names.index(p)] for y in years] for p in pick
        }
        return ChartSpec(
            chart_type="line", title=_title("line", lang), x_title=_t("year", lang, "axis"),
            y_title=_axis("value_bbl", lang), categories=cats,
            series={k2: v for k2, v in series.items()}, style=style, unit="$/bbl", seed=seed, lang=lang,
        )

    if chart_type == "grouped_bar":
        k = min(n_series or rng.randint(1, 3), len(VESSEL_RANGES))
        classes = rng.sample(list(VESSEL_RANGES), k)
        series = {
            _t(c, lang): _series_values(rng, n, *VESSEL_RANGES[c], trend=rng.uniform(-0.03, 0.04))
            for c in classes
        }
        return ChartSpec(
            chart_type="grouped_bar", title=_title("grouped_bar", lang),
            x_title=_t("year", lang, "axis"), y_title=_axis("value_day", lang), categories=cats,
            series=series, style=style, unit="$000/day", seed=seed, lang=lang,
        )

    if chart_type == "stacked_bar":
        k = min(n_series or rng.randint(2, 4), len(ORIGIN_RANGES))
        origins = rng.sample(list(ORIGIN_RANGES), k)
        series = {
            _t(o, lang): _series_values(rng, n, *ORIGIN_RANGES[o], trend=rng.uniform(-0.02, 0.03))
            for o in origins
        }
        return ChartSpec(
            chart_type="stacked_bar", title=_title("stacked_bar", lang),
            x_title=_t("year", lang, "axis"), y_title=_axis("value_mbd", lang), categories=cats,
            series=series, style=style, unit="mb/d", seed=seed, lang=lang,
        )

    if chart_type == "combo":
        vlcc = _series_values(rng, n, *VESSEL_RANGES["vlcc"], trend=rng.uniform(-0.03, 0.04))
        brent_line = [brent[y][0] for y in years]
        bar_name, line_name = _t("vlcc", lang), _t("brent", lang)
        secondary_title = _axis("value_bbl", lang)
        return ChartSpec(
            chart_type="combo", title=_title("combo", lang), x_title=_t("year", lang, "axis"),
            y_title=_axis("value_day", lang), categories=cats,
            series={bar_name: vlcc, line_name: brent_line}, style=style, unit="$000/day",
            secondary=[line_name], secondary_title=secondary_title, seed=seed, lang=lang,
        )

    if chart_type == "donut":
        n = min(n_categories or rng.choice([3, 4]), len(VESSEL_RANGES))
        classes = rng.sample(list(VESSEL_RANGES), n)
        raw = [rng.uniform(5, 40) for _ in classes]
        total = sum(raw)
        shares = [round(100 * r / total, 1) for r in raw]
        return ChartSpec(
            chart_type="donut", title=_title("donut", lang), x_title=_t("class", lang, "axis"),
            y_title=_axis("value_pct", lang), categories=[_t(c, lang) for c in classes],
            series={"Share": shares}, style=style, unit="%", seed=seed, lang=lang,
        )

    if chart_type == "waterfall":
        n = min(n_categories or rng.choice([4, 6]), len(DRIVER_RANGES) + 2)
        drivers = rng.sample(list(DRIVER_RANGES), n - 2)
        values = [round(rng.uniform(200, 800), 1)]
        for d in drivers:
            values.append(round(rng.uniform(*DRIVER_RANGES[d]), 1))
        end = round(values[0] + sum(values[1:]), 1)
        cats = [_t("start", lang)] + [_t(d, lang) for d in drivers] + [_t("end", lang)]
        return ChartSpec(
            chart_type="waterfall", title=_title("waterfall", lang), x_title=_t("cost_driver", lang, "axis"),
            y_title=_axis("value_t", lang), categories=cats, series={"Value": values + [end]},
            style=style, unit="$/t", seed=seed, lang=lang,
        )

    raise ValueError(f"chart type {chart_type!r} has no shipping template")
