"""Synthetic financial chart generator.

A chart is a ChartSpec: exact data, labels, and a Style. The same spec is rendered by
matplotlib, seaborn or plotly, so the ground-truth label (`to_chartcsv`) depends only on
the data, not on how the chart is drawn.

Chart types: grouped_bar, stacked_bar, line, combo (bar + line on two axes), donut, waterfall.

Style axes (sampled by `random_style`):
  palette        6 colour palettes
  legend         none | right | top | bottom
  orientation    vertical | horizontal (grouped and stacked bars only)
  value_labels   print the value on each bar
  grid           gridlines on/off
  figsize        wide | square | panoramic | tall
  font_scale     small | medium | large
  show_title     chart title on/off
  show_axis_titles  axis titles on/off
  background     white | ivory | light_gray
Data shape (`random_config`): number of bars/slices/steps and number of series.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ChartType = Literal["grouped_bar", "stacked_bar", "line", "combo", "donut", "waterfall"]
Backend = Literal["matplotlib", "seaborn", "plotly"]
Legend = Literal["none", "right", "top", "bottom"]

CHART_TYPES: tuple[ChartType, ...] = (
    "grouped_bar",
    "stacked_bar",
    "line",
    "combo",
    "donut",
    "waterfall",
)
BAR_TYPES = ("grouped_bar", "stacked_bar", "combo", "waterfall")
BACKENDS: tuple[Backend, ...] = ("matplotlib", "seaborn", "plotly")

PERIODS_KIND = "period"  # labels like "Q1 FY24"
YEARS_KIND = "year"  # labels like "2019"
SEGMENTS = ["Cloud", "Hardware", "Services", "Licensing", "Consumer", "Enterprise", "Retail", "Other"]
DRIVERS = ["Volume", "Price", "Mix", "Costs", "FX", "Other", "Tax", "Interest", "Divestments",
           "Impairments"]
METRICS = ["Revenue", "Gross profit", "Operating income", "Net income", "Free cash flow"]

PALETTES: dict[str, list[str]] = {
    "blue": ["#1f4e79", "#2e75b6", "#9dc3e6", "#a5a5a5", "#5b9bd5", "#bdd7ee", "#44546a", "#d6dce4"],
    "green": ["#0b3d2e", "#2e8b57", "#98d8a8", "#6c757d", "#52b788", "#b7e4c7", "#40916c", "#adb5bd"],
    "purple": ["#5b2c6f", "#8e44ad", "#d2b4de", "#7f8c8d", "#a569bd", "#e8daef", "#6c3483", "#aab7b8"],
    "red": ["#7b241c", "#c0392b", "#f5b7b1", "#566573", "#e74c3c", "#fadbd8", "#922b21", "#808b96"],
    "vivid": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    "gray": ["#212529", "#495057", "#868e96", "#ced4da", "#343a40", "#adb5bd", "#6c757d", "#dee2e6"],
}
FIGSIZES: dict[str, tuple[float, float]] = {
    "wide": (8, 5), "square": (6, 6), "panoramic": (10, 4.5), "tall": (6, 7),
}
PLOTLY_SIZES: dict[str, tuple[int, int]] = {
    "wide": (800, 500), "square": (600, 600), "panoramic": (1000, 450), "tall": (600, 700),
}
FONT_SCALES: dict[str, float] = {"small": 0.85, "medium": 1.0, "large": 1.25}
BACKGROUNDS: dict[str, str] = {"white": "#ffffff", "ivory": "#fbf8f1", "light_gray": "#f2f2f2"}
RENDER_DPIS: tuple[int, ...] = (72, 110, 150, 220)  # low-res PDF crops up to high-res scans
LEGENDS: tuple[Legend, ...] = ("none", "right", "top", "bottom")


@dataclass(frozen=True)
class Style:
    palette: str = "blue"
    legend: Legend = "right"
    orientation: str = "vertical"
    value_labels: bool = False
    grid: bool = True
    figsize: str = "wide"
    font_scale: str = "medium"
    show_title: bool = True
    show_axis_titles: bool = True
    background: str = "white"
    render_dpi: int = 150

    @property
    def colors(self) -> list[str]:
        return PALETTES[self.palette]

    @property
    def fig_inches(self) -> tuple[float, float]:
        return FIGSIZES[self.figsize]

    @property
    def font_factor(self) -> float:
        return FONT_SCALES[self.font_scale]

    @property
    def bg(self) -> str:
        return BACKGROUNDS[self.background]


def random_style(chart_type: ChartType, rng: random.Random) -> Style:
    """Sample one style that is valid for this chart type."""
    if chart_type == "waterfall":
        legend_choices: tuple[Legend, ...] = ("none",)
    elif chart_type == "donut":
        legend_choices = ("none", "right")
    else:
        legend_choices = LEGENDS
    orientation = "vertical"
    if chart_type in ("grouped_bar", "stacked_bar") and rng.random() < 0.4:
        orientation = "horizontal"
    return Style(
        palette=rng.choice(list(PALETTES)),
        legend=rng.choice(legend_choices),
        orientation=orientation,
        value_labels=rng.random() < 0.5,
        grid=rng.random() < 0.6,
        figsize=rng.choice(list(FIGSIZES)),
        font_scale=rng.choice(list(FONT_SCALES)),
        show_title=rng.random() < 0.85,
        show_axis_titles=rng.random() < 0.8,
        background=rng.choice(list(BACKGROUNDS)),
        render_dpi=rng.choice(RENDER_DPIS),
    )


def random_config(chart_type: ChartType, rng: random.Random) -> dict:
    """Sample bar/series counts and a style for one chart."""
    if chart_type in ("grouped_bar", "stacked_bar", "line"):
        n_cat, n_ser = rng.choice([3, 5, 8, 12]), rng.randint(1, 4)
    elif chart_type == "combo":
        n_cat, n_ser = rng.choice([3, 5, 8, 12]), 2
    elif chart_type == "donut":
        n_cat, n_ser = rng.choice([3, 5, 8]), 1
    else:  # waterfall
        n_cat, n_ser = rng.choice([4, 6, 8, 12]), 1
    return {"n_categories": n_cat, "n_series": n_ser, "style": random_style(chart_type, rng)}


@dataclass
class ChartSpec:
    chart_type: ChartType
    title: str
    x_title: str
    y_title: str
    categories: list[str]
    series: dict[str, list[float]]  # series name -> values aligned with categories
    style: Style = field(default_factory=Style)
    unit: str = ""  # e.g. "$m", "%"
    secondary: list[str] = field(default_factory=list)  # series on the right axis (combo)
    seed: int = 0

    def to_chartcsv(self) -> str:
        """Ground-truth label: a short header block followed by the data table as CSV."""
        lines = [
            f"type: {self.chart_type}",
            f"title: {self.title}",
            f"x_axis: {self.x_title}",
            f"y_axis: {self.y_title}",
            f"unit: {self.unit or 'none'}",
            "data:",
            ",".join([self.x_title] + list(self.series)),
        ]
        for i, cat in enumerate(self.categories):
            row = [cat] + [_fmt(self.series[s][i]) for s in self.series]
            lines.append(",".join(row))
        return "\n".join(lines)

    def meta(self) -> dict:
        return {"chart_type": self.chart_type, "n_categories": len(self.categories),
                "n_series": len(self.series), **asdict(self.style)}


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))


def _labels(kind: str, n: int) -> list[str]:
    if kind == PERIODS_KIND:
        return [f"Q{i % 4 + 1} FY{24 + i // 4}" for i in range(n)]
    start = 2024 - n + 1
    return [str(start + i) for i in range(n)]


def _series_values(rng: random.Random, n: int, lo: float, hi: float, trend: float) -> list[float]:
    base = rng.uniform(lo, hi)
    return [
        round(max(0.0, base * (1 + trend * i) * (1 + rng.uniform(-0.08, 0.08))), 1)
        for i in range(n)
    ]


# ---------------------------------------------------------------- data generation

def generate_spec(
    chart_type: ChartType,
    seed: int,
    n_categories: int | None = None,
    n_series: int | None = None,
    style: Style | None = None,
) -> ChartSpec:
    """Build one financial chart with reproducible random data.

    Any argument left as None is sampled from `seed`.
    """
    rng = random.Random(seed)
    metric = rng.choice(METRICS)
    unit = rng.choice(["$m", "$bn", "%"])
    if style is None:
        style = random_style(chart_type, rng)

    if chart_type in ("grouped_bar", "stacked_bar", "line"):
        n = n_categories or rng.choice([3, 5, 8, 12])
        k = n_series or rng.randint(1, 4)
        kind = rng.choice([PERIODS_KIND, YEARS_KIND])
        names = rng.sample(SEGMENTS, k)
        trend = rng.uniform(-0.02, 0.08)
        series = {name: _series_values(rng, n, 50, 900, trend) for name in names}
        if unit == "%":
            series = {nm: [min(v / 10, 99.0) for v in vals] for nm, vals in series.items()}
        return ChartSpec(
            chart_type=chart_type,
            title=f"{metric} by {'segment' if k > 1 else 'period'}",
            x_title="Period" if kind == PERIODS_KIND else "Year",
            y_title=f"{metric} ({unit})",
            categories=_labels(kind, n),
            series=series,
            style=style,
            unit=unit,
            seed=seed,
        )

    if chart_type == "combo":
        n = n_categories or rng.choice([3, 5, 8, 12])
        kind = rng.choice([PERIODS_KIND, YEARS_KIND])
        bar_name, line_name = rng.sample(["Revenue", "Gross profit", "Units sold"], 2)
        series = {
            bar_name: _series_values(rng, n, 200, 1200, 0.05),
            line_name: _series_values(rng, n, 20, 45, 0.01),  # margin-style %
        }
        return ChartSpec(
            chart_type="combo",
            title=f"{bar_name} and {line_name} over time",
            x_title="Period" if kind == PERIODS_KIND else "Year",
            y_title=f"{bar_name} ({unit})",
            categories=_labels(kind, n),
            series=series,
            style=style,
            unit=unit,
            secondary=[line_name],
            seed=seed,
        )

    if chart_type == "donut":
        n = n_categories or rng.choice([3, 5, 8])
        names = rng.sample(SEGMENTS, n)
        raw = [rng.uniform(5, 40) for _ in names]
        total = sum(raw)
        shares = [round(100 * r / total, 1) for r in raw]
        return ChartSpec(
            chart_type="donut",
            title=f"{metric} mix by segment (%)",
            x_title="Segment",
            y_title="Share (%)",
            categories=names,
            series={"Share": shares},
            style=style,
            unit="%",
            seed=seed,
        )

    if chart_type == "waterfall":
        n = n_categories or rng.choice([4, 6, 8, 12])
        drivers = rng.sample(DRIVERS, n - 2)
        values = [round(rng.uniform(200, 800), 1)]
        for _ in drivers:
            values.append(round(rng.uniform(-150, 150), 1))
        end = round(values[0] + sum(values[1:]), 1)
        return ChartSpec(
            chart_type="waterfall",
            title=f"{metric} bridge",
            x_title="Driver",
            y_title=f"Change ({unit})",
            categories=["Start"] + drivers + ["End"],
            series={"Value": values + [end]},
            style=style,
            unit=unit,
            seed=seed,
        )

    raise ValueError(f"unknown chart type {chart_type!r}")


# ---------------------------------------------------------------- renderers

def render(spec: ChartSpec, backend: Backend, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "matplotlib":
        _render_matplotlib(spec, path)
    elif backend == "seaborn":
        _render_seaborn(spec, path)
    elif backend == "plotly":
        _render_plotly(spec, path)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return path


def _is_horizontal(spec: ChartSpec) -> bool:
    return spec.style.orientation == "horizontal" and spec.chart_type in ("grouped_bar", "stacked_bar")


def _mpl_bar(ax, pos, vals, width, bottom, color, label, horizontal):
    if horizontal:
        return ax.barh(pos, vals, width, left=bottom, color=color, label=label)
    return ax.bar(pos, vals, width, bottom=bottom, color=color, label=label)


def _mpl_axes(ax, spec: ChartSpec, horizontal: bool) -> None:
    """Category ticks, axis titles and grid for matplotlib/seaborn axes."""
    import numpy as np

    cats = spec.categories
    pos = np.arange(len(cats))
    st = spec.style
    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(cats)
        ax.grid(axis="x", alpha=0.3 if st.grid else 0)
        ax.grid(axis="y", visible=False)
        value_label, cat_label = spec.y_title, spec.x_title
        if st.show_axis_titles:
            ax.set_xlabel(value_label)
            ax.set_ylabel(cat_label)
    else:
        ax.set_xticks(pos)
        long = len(cats) >= 7
        ax.set_xticklabels(cats, rotation=30 if long else 0, ha="right" if long else "center")
        ax.grid(axis="y", alpha=0.3 if st.grid else 0)
        ax.grid(axis="x", visible=False)
        if st.show_axis_titles:
            ax.set_xlabel(spec.x_title)
            ax.set_ylabel(spec.y_title)


def _mpl_place_legend(ax, pos: Legend, others=()) -> None:
    """Put one legend on `ax` covering `ax` and any twin axes in `others`."""
    handles, labels = ax.get_legend_handles_labels()
    for other in others:
        h, lab = other.get_legend_handles_labels()
        handles, labels = handles + h, labels + lab
        if other.get_legend():
            other.get_legend().remove()
    if ax.get_legend():
        ax.get_legend().remove()
    if pos == "none" or not handles:
        return
    if pos == "right":
        kw = dict(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    elif pos == "top":
        kw = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(labels), 4))
    else:
        kw = dict(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(len(labels), 4))
    ax.legend(handles, labels, frameon=False, **kw)


def _mpl_value_labels(ax, horizontal: bool = False) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=7)


def _draw_waterfall(ax, spec: ChartSpec) -> None:
    """Floating-bar waterfall: first and last bars are totals, middle bars float on the running sum."""
    vals = spec.series["Value"]
    colors = spec.style.colors
    running = 0.0
    for i, v in enumerate(vals):
        if i == 0:
            ax.bar(i, v, 0.6, color=colors[0])
            running = v
        elif i == len(vals) - 1:
            ax.bar(i, v, 0.6, color=colors[0])
        else:
            ax.bar(i, v, 0.6, bottom=running, color=colors[1] if v >= 0 else colors[3])
            running += v
    ax.axhline(0, color="black", linewidth=0.8)


def _draw_donut(ax, spec: ChartSpec) -> None:
    colors = spec.style.colors
    shares = spec.series["Share"]
    cycle = [colors[i % len(colors)] for i in range(len(shares))]
    labels_on_slices = spec.style.legend == "none"
    wedges, _, _ = ax.pie(
        shares,
        labels=spec.categories if labels_on_slices else None,
        colors=cycle,
        startangle=90,
        wedgeprops={"width": 0.4},
        autopct="%1.1f%%",
    )
    for w, cat in zip(wedges, spec.categories):
        w.set_label(cat)
    ax.axis("equal")


def _finish_mpl(fig, ax, spec: ChartSpec, path: Path, twin=()) -> None:
    st = spec.style
    if spec.chart_type != "donut":
        _mpl_axes(ax, spec, _is_horizontal(spec))
    if st.show_title:
        ax.set_title(spec.title)
    _mpl_place_legend(ax, st.legend, twin)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=st.render_dpi)


def _render_matplotlib(spec: ChartSpec, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    st = spec.style
    horizontal = _is_horizontal(spec)
    base = 10 * st.font_factor
    with plt.rc_context({"font.size": base, "axes.titlesize": base * 1.3,
                         "axes.labelsize": base * 1.05}):
        fig, ax = plt.subplots(figsize=st.fig_inches, dpi=st.render_dpi)
        fig.patch.set_facecolor(st.bg)
        ax.set_facecolor(st.bg)
        colors = st.colors
        names = [n for n in spec.series if n not in spec.secondary]
        pos = np.arange(len(spec.categories))
        twin: list = []

        if spec.chart_type == "grouped_bar":
            w = 0.8 / len(names)
            for i, n in enumerate(names):
                _mpl_bar(ax, pos + (i - (len(names) - 1) / 2) * w, spec.series[n], w, 0,
                         colors[i % len(colors)], n, horizontal)
        elif spec.chart_type == "stacked_bar":
            bottom = np.zeros(len(pos))
            for i, n in enumerate(names):
                vals = np.array(spec.series[n])
                _mpl_bar(ax, pos, vals, 0.6, bottom, colors[i % len(colors)], n, horizontal)
                bottom = bottom + vals
        elif spec.chart_type == "line":
            for i, n in enumerate(names):
                ax.plot(pos, spec.series[n], marker="o", label=n, color=colors[i % len(colors)])
        elif spec.chart_type == "combo":
            ax.bar(pos, spec.series[names[0]], 0.6, label=names[0], color=colors[0])
            ax2 = ax.twinx()
            ax2.plot(pos, spec.series[spec.secondary[0]], marker="o", color=colors[1],
                     label=spec.secondary[0])
            if st.show_axis_titles:
                ax2.set_ylabel(f"{spec.secondary[0]} (%)")
            twin = [ax2]
        elif spec.chart_type == "donut":
            _draw_donut(ax, spec)
        elif spec.chart_type == "waterfall":
            _draw_waterfall(ax, spec)

        if st.value_labels and spec.chart_type in BAR_TYPES:
            _mpl_value_labels(ax)
        _finish_mpl(fig, ax, spec, path, twin)
        plt.close(fig)


def _render_seaborn(spec: ChartSpec, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    st = spec.style
    horizontal = _is_horizontal(spec)
    colors = st.colors
    sns.set_theme(
        style="whitegrid" if st.grid else "white",
        font_scale=st.font_factor,
        rc={"figure.facecolor": st.bg, "axes.facecolor": st.bg},
    )
    names = [n for n in spec.series if n not in spec.secondary]
    palette = [colors[i % len(colors)] for i in range(len(names))]
    long = pd.DataFrame([
        {spec.x_title: c, "series": n, "value": spec.series[n][i]}
        for n in names
        for i, c in enumerate(spec.categories)
    ])
    fig, ax = plt.subplots(figsize=st.fig_inches, dpi=st.render_dpi)
    twin: list = []
    if spec.chart_type == "grouped_bar":
        if horizontal:
            sns.barplot(data=long, x="value", y=spec.x_title, hue="series", palette=palette,
                        orient="h", ax=ax)
        else:
            sns.barplot(data=long, x=spec.x_title, y="value", hue="series", palette=palette, ax=ax)
    elif spec.chart_type == "stacked_bar":
        pivot = long.pivot(index=spec.x_title, columns="series", values="value").loc[spec.categories]
        pivot.plot(kind="barh" if horizontal else "bar", stacked=True, ax=ax, color=palette,
                   rot=0, width=0.6, legend=False)
    elif spec.chart_type == "line":
        sns.lineplot(data=long, x=spec.x_title, y="value", hue="series", palette=palette,
                     marker="o", ax=ax)
    elif spec.chart_type == "combo":
        sns.barplot(data=long, x=spec.x_title, y="value", hue="series", palette=palette,
                    ax=ax, legend=False)
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(spec.categories)), spec.series[spec.secondary[0]], marker="o",
                 color=colors[1], label=spec.secondary[0])
        twin = [ax2]
    elif spec.chart_type == "donut":
        _draw_donut(ax, spec)
    elif spec.chart_type == "waterfall":
        _draw_waterfall(ax, spec)

    if st.value_labels and spec.chart_type in BAR_TYPES:
        _mpl_value_labels(ax)
    if spec.chart_type != "donut":
        _mpl_axes(ax, spec, horizontal)
    if st.show_title:
        ax.set_title(spec.title)
    _mpl_place_legend(ax, st.legend, twin)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=st.render_dpi)
    plt.close(fig)


def _render_plotly(spec: ChartSpec, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st = spec.style
    horizontal = _is_horizontal(spec)
    colors = st.colors
    cats = spec.categories
    names = [n for n in spec.series if n not in spec.secondary]
    color_of = lambda i: colors[i % len(colors)]  # noqa: E731

    if spec.chart_type == "combo":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=cats, y=spec.series[names[0]], name=names[0], marker_color=colors[0],
                    secondary_y=False)
        fig.add_scatter(x=cats, y=spec.series[spec.secondary[0]], name=spec.secondary[0],
                        mode="lines+markers", marker_color=colors[1], secondary_y=True)
        fig.update_yaxes(title_text=spec.y_title, secondary_y=False)
        fig.update_yaxes(title_text=f"{spec.secondary[0]} (%)", secondary_y=True)
    elif spec.chart_type == "donut":
        fig = go.Figure(go.Pie(
            labels=cats, values=spec.series["Share"], hole=0.4,
            marker=dict(colors=[color_of(i) for i in range(len(cats))]),
            textinfo="percent", showlegend=st.legend != "none",
        ))
    elif spec.chart_type == "waterfall":
        vals = spec.series["Value"]
        measure = ["absolute"] + ["relative"] * (len(vals) - 2) + ["total"]
        fig = go.Figure(go.Waterfall(
            x=cats, y=vals, measure=measure, showlegend=False,
            increasing=dict(marker=dict(color=colors[1])),
            decreasing=dict(marker=dict(color=colors[3])),
            totals=dict(marker=dict(color=colors[0])),
        ))
    else:
        fig = go.Figure()
        for i, n in enumerate(names):
            if spec.chart_type == "line":
                fig.add_scatter(x=cats, y=spec.series[n], name=n, mode="lines+markers",
                                marker_color=color_of(i))
            elif horizontal:
                fig.add_bar(x=spec.series[n], y=cats, orientation="h", name=n,
                            marker_color=color_of(i))
            else:
                fig.add_bar(x=cats, y=spec.series[n], name=n, marker_color=color_of(i))
        if spec.chart_type == "stacked_bar":
            fig.update_layout(barmode="stack")
        elif spec.chart_type == "grouped_bar":
            fig.update_layout(barmode="group")
        if horizontal:
            fig.update_xaxes(title_text=spec.y_title if st.show_axis_titles else None)
            fig.update_yaxes(title_text=spec.x_title if st.show_axis_titles else None)
        else:
            fig.update_xaxes(title_text=spec.x_title if st.show_axis_titles else None)
            fig.update_yaxes(title_text=spec.y_title if st.show_axis_titles else None)

    if st.value_labels and spec.chart_type in ("grouped_bar", "stacked_bar", "combo"):
        fmt = "%{x:.0f}" if horizontal else "%{y:.0f}"
        fig.update_traces(texttemplate=fmt, textposition="outside", selector=dict(type="bar"))
    if st.value_labels and spec.chart_type == "waterfall":
        fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")

    legend = st.legend
    if spec.chart_type == "donut":
        pass  # showlegend already set on the pie; position below
    elif spec.chart_type == "waterfall" or legend == "none":
        fig.update_layout(showlegend=False)
    else:
        fig.update_layout(showlegend=True)
        if legend == "right":
            fig.update_layout(legend=dict(x=1.02, y=0.5, yanchor="middle"))
        elif legend == "top":
            fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12,
                                          yanchor="bottom"))
        else:
            fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25,
                                          yanchor="top"))

    width, height = PLOTLY_SIZES[st.figsize]
    fig.update_layout(
        title=spec.title if st.show_title else None,
        template="plotly_white" if st.grid else "simple_white",
        width=width, height=height,
        font=dict(size=12 * st.font_factor),
        paper_bgcolor=st.bg, plot_bgcolor=st.bg,
        margin=dict(t=80 if st.show_title else 30, b=110 if legend == "bottom" else 60),
    )
    if spec.chart_type != "donut":
        fig.update_yaxes(showgrid=st.grid and not horizontal)
        fig.update_xaxes(showgrid=st.grid and horizontal)
    # plotly scale 1.0 = 800px wide; scale by dpi relative to the 150 dpi baseline
    fig.write_image(str(path), scale=st.render_dpi / 150)
