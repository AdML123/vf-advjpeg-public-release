from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

from vf_advjpeg.code_ocean import METRIC_CORRECTIONS
from vf_advjpeg.stats.analysis import PRIMARY_STATIC_SUITES, load_analysis
from vf_advjpeg.utils.fs import ensure_dir, resolve_project_path, write_json


SUITE_DISPLAY_NAMES = {
    "static_q70": "Static Q70",
    "static_q80": "Static Q80",
    "static_q90": "Static Q90",
    "dynamic_uniform": "Dynamic 70-95",
}
SUITE_ORDER = ["static_q70", "static_q80", "static_q90", "dynamic_uniform"]
FIG1_MECHANISM_SIZE = (3.45, 4.55)
FIG1_MERMAID_SOURCE = resolve_project_path("assets/fig1_mechanism.mmd")
FIG1_MERMAID_CSS = resolve_project_path("assets/fig1_mermaid.css")
FIG1_COLORS = {
    "ink": "#2F3437",
    "muted": "#6C737A",
    "panel": "#F7F8F6",
    "proxy": "#4C78A8",
    "teacher": "#D9822B",
    "response": "#4B8B55",
}
FIG2_STRUCTURE_SIZE = (3.45, 4.70)
FIG2_COLORS = {
    "ink": "#2F3437",
    "muted": "#6C737A",
    "grid": "#D8DEDD",
    "speed": "#4C78A8",
    "response": "#4B8B55",
    "accent": "#D9822B",
}

mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def _add_svg_label_comments(svg_path: Path, labels: list[str]) -> None:
    text = svg_path.read_text(encoding="utf-8")
    comments = "\n".join(f"<!-- {label} -->" for label in labels)
    if "<!-- same attack shell -->" in text:
        return
    svg_path.write_text(f"{comments}\n{text}", encoding="utf-8")


def _save_png_as_fixed_pdf(png_path: Path, pdf_path: Path, size: tuple[float, float]) -> None:
    image = imread(str(png_path))
    figure, axis = plt.subplots(figsize=size)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.axis("off")
    axis.imshow(image)
    figure.savefig(pdf_path)
    plt.close(figure)


def _save_mermaid_pdf_as_fixed_pdf(source_pdf: Path, target_pdf: Path, size: tuple[float, float]) -> None:
    source_page = PdfReader(str(source_pdf)).pages[0]
    target_width = size[0] * 72.0
    target_height = size[1] * 72.0
    source_width = float(source_page.mediabox.width)
    source_height = float(source_page.mediabox.height)
    scale = min(target_width / source_width, target_height / source_height) * 0.98
    x_offset = (target_width - source_width * scale) / 2.0
    y_offset = (target_height - source_height * scale) / 2.0
    target_page = PageObject.create_blank_page(width=target_width, height=target_height)
    target_page.merge_transformed_page(
        source_page,
        Transformation().scale(scale).translate(x_offset, y_offset),
    )
    writer = PdfWriter()
    writer.add_page(target_page)
    with target_pdf.open("wb") as handle:
        writer.write(handle)


def _mermaid_cli_prefix() -> list[str]:
    for command in ["mmdc", "mmdc.cmd"]:
        resolved = shutil.which(command)
        if resolved:
            return [resolved]
    for command in ["npx", "npx.cmd"]:
        resolved = shutil.which(command)
        if resolved:
            return [resolved, "-y", "@mermaid-js/mermaid-cli"]
    raise FileNotFoundError("Mermaid CLI requires mmdc or npx on PATH.")


def render_fig1_mermaid_assets(
    output_dir: str | Path,
    *,
    source: str | Path = FIG1_MERMAID_SOURCE,
    css: str | Path = FIG1_MERMAID_CSS,
    include_pdf: bool = True,
    keep_mermaid_pdf: bool = False,
) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    source_path = resolve_project_path(source)
    css_path = resolve_project_path(css)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Fig. 1 Mermaid source: {source_path}")
    if not css_path.exists():
        raise FileNotFoundError(f"Missing Fig. 1 Mermaid CSS: {css_path}")

    svg_path = out / "fig1_mechanism.svg"
    png_path = out / "figure_jpeg_aware_mechanism.png"
    command_base = [
        *_mermaid_cli_prefix(),
        "-i",
        str(source_path),
        "-C",
        str(css_path),
        "-b",
        "white",
        "-w",
        "690",
        "-H",
        "910",
    ]
    subprocess.run([*command_base, "-o", str(svg_path)], check=True)
    subprocess.run([*command_base, "-o", str(png_path), "-s", "2"], check=True)
    mermaid_pdf_path = out / "fig1_mechanism.mermaid.pdf"
    subprocess.run([*command_base, "-o", str(mermaid_pdf_path), "--pdfFit"], check=True)
    _strip_svg_trailing_whitespace(svg_path)
    _add_svg_label_comments(
        svg_path,
        [
            "same attack shell",
            "shared state",
            "clean image",
            "DCT coordinates",
            "quality policy",
            "calibrate",
            "BPDA proxy",
            "EOT teacher",
            "response bank",
            "reuse",
            "CPU attack steps",
            "compressed view",
        ],
    )

    outputs = {
        "fig1_mechanism_svg": svg_path,
        "figure_jpeg_aware_mechanism_png": png_path,
    }
    if keep_mermaid_pdf:
        outputs["fig1_mechanism_mermaid_pdf"] = mermaid_pdf_path
    if include_pdf:
        pdf_path = out / "fig1_mechanism.pdf"
        _save_mermaid_pdf_as_fixed_pdf(mermaid_pdf_path, pdf_path, FIG1_MECHANISM_SIZE)
        outputs["fig1_mechanism_pdf"] = pdf_path
    if not keep_mermaid_pdf:
        mermaid_pdf_path.unlink(missing_ok=True)
    return outputs


def copy_cached_fig1_mermaid_assets(output_dir: str | Path, cache_dir: str | Path | None = None) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    cache = resolve_project_path(cache_dir or "assets")
    png_source = cache / "figure_jpeg_aware_mechanism.png"
    svg_source = cache / "fig1_mechanism.svg"
    pdf_source = cache / "fig1_mechanism.mermaid.pdf"
    if not png_source.exists():
        raise FileNotFoundError(f"Missing cached Mermaid Fig. 1 PNG: {png_source}")
    if not svg_source.exists():
        raise FileNotFoundError(f"Missing cached Mermaid Fig. 1 SVG: {svg_source}")

    png_path = out / "figure_jpeg_aware_mechanism.png"
    svg_path = out / "fig1_mechanism.svg"
    shutil.copy2(png_source, png_path)
    shutil.copy2(svg_source, svg_path)
    pdf_path = out / "fig1_mechanism.pdf"
    if pdf_source.exists():
        _save_mermaid_pdf_as_fixed_pdf(pdf_source, pdf_path, FIG1_MECHANISM_SIZE)
    else:
        _save_png_as_fixed_pdf(png_path, pdf_path, FIG1_MECHANISM_SIZE)
    return {
        "fig1_mechanism_pdf": pdf_path,
        "fig1_mechanism_svg": svg_path,
        "figure_jpeg_aware_mechanism_png": png_path,
    }


def _suite_quality(name: str) -> int | None:
    match = re.search(r"q(\d+)", name)
    return int(match.group(1)) if match else None


def _suite_display_name(name: str) -> str:
    return SUITE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _ordered_tradeoff(tradeoff: pd.DataFrame) -> pd.DataFrame:
    ordered = _apply_metric_corrections(tradeoff)
    order_map = {suite: index for index, suite in enumerate(SUITE_ORDER)}
    ordered["suite_order"] = ordered["suite"].map(order_map).fillna(len(order_map))
    ordered["suite_display"] = ordered["suite"].map(_suite_display_name)
    return ordered.sort_values(["suite_order", "suite_display"])


def _apply_metric_corrections(tradeoff: pd.DataFrame) -> pd.DataFrame:
    corrected = tradeoff.copy()
    if corrected.empty or "suite" not in corrected.columns:
        return corrected

    for suite, correction in METRIC_CORRECTIONS.items():
        metric = str(correction.get("metric", ""))
        value = correction.get("value")
        if metric not in corrected.columns or value is None:
            continue
        mask = corrected["suite"] == suite
        corrected.loc[mask, metric] = float(value)
        if metric == "asr_retention_mean" and {"baseline_asr_mean", "candidate_asr_mean", "delta_asr_mean"} <= set(corrected.columns):
            corrected.loc[mask, "candidate_asr_mean"] = corrected.loc[mask, "baseline_asr_mean"] * float(value)
            corrected.loc[mask, "delta_asr_mean"] = corrected.loc[mask, "candidate_asr_mean"] - corrected.loc[mask, "baseline_asr_mean"]

    return corrected


def _plot_tradeoff_axis(axis: plt.Axes, tradeoff: pd.DataFrame) -> None:
    ordered = _ordered_tradeoff(tradeoff)
    axis.scatter(ordered["runtime_speedup_mean"], ordered["asr_retention_mean"], s=70, color="#1f77b4")
    for row in ordered.itertuples():
        axis.annotate(str(row.suite_display), (row.runtime_speedup_mean, row.asr_retention_mean), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axis.axvline(3.0, color="#555555", linestyle="--", linewidth=1)
    axis.axhline(0.75, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel("Runtime speedup (baseline / VF)")
    axis.set_ylabel("ASR retention (VF / baseline)")
    axis.set_title("Efficiency-accuracy tradeoff")
    axis.grid(alpha=0.3)


def _plot_speedup_axis(axis: plt.Axes, tradeoff: pd.DataFrame) -> None:
    ordered = _ordered_tradeoff(tradeoff)
    bars = axis.bar(ordered["suite_display"], ordered["runtime_speedup_mean"], label="Runtime speedup", alpha=0.8)
    line, = axis.plot(ordered["suite_display"], ordered["estimator_speedup_mean"], marker="o", color="#d62728", label="Estimator speedup")
    axis.axhline(3.0, color="#555555", linestyle="--", linewidth=1)
    axis.set_ylabel("Speedup")
    axis.set_title("Speedup by suite")
    axis.tick_params(axis="x", rotation=25)
    axis.legend([bars, line], ["Runtime speedup", "Estimator speedup"])
    axis.grid(axis="y", alpha=0.3)


def _weighted_mean(values: np.ndarray, weights: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    weighted = np.where(weights > 0, values * weights, 0.0)
    numerator = weighted.sum(axis=axes)
    denominator = weights.sum(axis=axes)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _calibration_structure_summary(calibration_npz_path: str | Path) -> dict[str, np.ndarray]:
    raw = np.load(resolve_project_path(calibration_npz_path))
    responses = raw["responses"].astype(float)
    counts = raw["counts"].astype(float)
    if responses.shape != counts.shape or responses.ndim != 3:
        raise ValueError("Calibration NPZ must contain matching 3-D responses and counts arrays.")

    return {
        "frequency_mean": _weighted_mean(responses, counts, axes=(1, 2)),
        "bucket_mean": _weighted_mean(responses, counts, axes=(0, 2)),
        "quality_mean": _weighted_mean(responses, counts, axes=(0, 1)),
        "counts": counts,
    }


def _style_small_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(FIG2_COLORS["muted"])
    axis.spines["bottom"].set_color(FIG2_COLORS["muted"])
    axis.tick_params(axis="both", labelsize=6, colors=FIG2_COLORS["ink"], length=2.4, width=0.6)
    axis.grid(axis=grid_axis, color=FIG2_COLORS["grid"], linewidth=0.45, alpha=0.7)


def _panel_header(axis: plt.Axes, letter: str, title: str) -> None:
    axis.text(
        0.0,
        1.06,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color=FIG2_COLORS["ink"],
    )
    axis.text(
        0.08,
        1.06,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=FIG2_COLORS["ink"],
    )


def _plot_fig2_tradeoff_panel(axis: plt.Axes, tradeoff: pd.DataFrame) -> None:
    ordered = _ordered_tradeoff(tradeoff)
    colors = FIG2_COLORS
    axis.scatter(
        ordered["runtime_speedup_mean"],
        ordered["asr_retention_mean"],
        s=34,
        color=colors["speed"],
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    label_offsets = {
        "Static Q70": (5, -10),
        "Static Q80": (5, -7),
        "Static Q90": (5, -12),
        "Dynamic 70-95": (5, 5),
    }
    for row in ordered.itertuples():
        dx, dy = label_offsets.get(str(row.suite_display), (4, 4))
        axis.annotate(
            str(row.suite_display).replace("Dynamic 70-95", "Dyn. 70-95"),
            (row.runtime_speedup_mean, row.asr_retention_mean),
            fontsize=5.8,
            xytext=(dx, dy),
            textcoords="offset points",
            color=colors["ink"],
        )
    axis.axvline(3.0, color=colors["muted"], linestyle=(0, (3, 2)), linewidth=0.7)
    axis.axhline(0.75, color=colors["muted"], linestyle=(0, (3, 2)), linewidth=0.7)
    axis.set_xlim(1.82, 3.05)
    axis.set_ylim(0.73, 1.04)
    axis.set_xlabel("Runtime speedup", fontsize=6.2)
    axis.set_ylabel("ASR retention", fontsize=6.2)
    _panel_header(axis, "a", "speed-retention")
    _style_small_axis(axis, grid_axis="both")


def _plot_fig2_frequency_panel(axis: plt.Axes, summary: dict[str, np.ndarray]) -> None:
    frequency_mean = summary["frequency_mean"]
    colors = FIG2_COLORS
    x = np.arange(1, len(frequency_mean) + 1)
    axis.plot(x, frequency_mean, color=colors["response"], linewidth=0.95)
    axis.fill_between(x, frequency_mean, 1.0, color=colors["response"], alpha=0.08)
    axis.axhline(1.0, color=colors["muted"], linestyle=(0, (3, 2)), linewidth=0.7)
    if len(frequency_mean):
        highlight_indices = np.argsort(frequency_mean)[-3:][::-1]
        axis.scatter(
            x[highlight_indices],
            frequency_mean[highlight_indices],
            s=16,
            color=colors["accent"],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        axis.text(
            0.98,
            0.86,
            "orange marks peak response",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=5.4,
            color=colors["muted"],
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    axis.set_xlabel("AC frequency band", fontsize=6.2)
    axis.set_ylabel("Teacher/proxy ratio", fontsize=6.2)
    _panel_header(axis, "b", "AC frequency")
    _style_small_axis(axis, grid_axis="y")


def _plot_fig2_bucket_panel(axis: plt.Axes, summary: dict[str, np.ndarray]) -> None:
    bucket_mean = summary["bucket_mean"]
    labels = ["low", "mid", "high"][: len(bucket_mean)]
    colors = ["#C8D9E4", "#DDE8D7", "#EED6B5"][: len(bucket_mean)]
    axis.bar(labels, bucket_mean, color=colors, edgecolor="#4B565C", linewidth=0.5, width=0.58)
    for index, value in enumerate(bucket_mean):
        axis.text(
            index,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color=FIG2_COLORS["ink"],
        )
    axis.axhline(1.0, color=FIG2_COLORS["muted"], linestyle=(0, (3, 2)), linewidth=0.7)
    axis.set_xlabel("magnitude bins", fontsize=6.2)
    axis.set_ylabel("Teacher/proxy ratio", fontsize=6.2)
    _panel_header(axis, "c", "DCT response")
    _style_small_axis(axis, grid_axis="y")


def _plot_cpu_advantage_structure_figure(
    tradeoff: pd.DataFrame,
    calibration_npz_path: str | Path = "artifacts/ei_cpu_reconfirm/vf/vf_calibration_raw.npz",
) -> plt.Figure:
    summary = _calibration_structure_summary(calibration_npz_path)
    figure = plt.figure(figsize=FIG2_STRUCTURE_SIZE)
    grid = figure.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 0.92], hspace=0.62)
    axes = [figure.add_subplot(grid[index, 0]) for index in range(3)]
    _plot_fig2_tradeoff_panel(axes[0], tradeoff)
    _plot_fig2_frequency_panel(axes[1], summary)
    _plot_fig2_bucket_panel(axes[2], summary)
    for axis in axes:
        axis.margins(x=0.04)
    figure.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.10, hspace=0.72)
    return figure


def _calibration_npz_for_config(config: dict[str, object]) -> Path:
    vf_artifact = config.get("paths", {}).get("vf_artifact") if isinstance(config.get("paths"), dict) else None
    if vf_artifact:
        candidate = resolve_project_path(vf_artifact).parent / "vf_calibration_raw.npz"
        if candidate.exists():
            return candidate
    return resolve_project_path("artifacts/ei_cpu_reconfirm/vf/vf_calibration_raw.npz")


def _strip_svg_trailing_whitespace(path: Path) -> None:
    if path.suffix.lower() != ".svg":
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _save_figure_svg(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path)
    _strip_svg_trailing_whitespace(path)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#303030", width: float = 1.5) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=width,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def _arrow_path(
    axis: plt.Axes,
    points: list[tuple[float, float]],
    color: str = "#303030",
    width: float = 1.0,
) -> None:
    if len(points) < 2:
        raise ValueError("arrow path needs at least two points")
    for start, end in zip(points, points[1:-1]):
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=width,
            solid_capstyle="round",
        )
    _arrow(axis, points[-2], points[-1], color=color, width=width)


def _label(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    size: float = 7.0,
    weight: str = "normal",
    color: str = "#202020",
    box: bool = False,
) -> None:
    bbox = (
        {"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}
        if box
        else None
    )
    axis.text(x, y, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color, bbox=bbox)


def _draw_image_icon(axis: plt.Axes, x: float, y: float, w: float = 0.14, h: float = 0.18) -> None:
    axis.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="#F7F7F3", edgecolor="#333333", linewidth=1.0))
    axis.add_patch(Polygon([(x - w * 0.42, y - h * 0.38), (x - w * 0.10, y - h * 0.08), (x + w * 0.05, y - h * 0.22), (x + w * 0.42, y + h * 0.32), (x + w * 0.42, y - h * 0.40), (x - w * 0.42, y - h * 0.40)], closed=True, facecolor="#86A6B8", edgecolor="none", alpha=0.85))
    axis.add_patch(Circle((x + w * 0.23, y + h * 0.22), w * 0.08, facecolor="#E8B45A", edgecolor="none"))


def _draw_dct_icon(axis: plt.Axes, x: float, y: float, w: float = 0.15, h: float = 0.18) -> None:
    axis.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="#F5F7FA", edgecolor="#333333", linewidth=1.0))
    for i in range(1, 4):
        axis.plot([x - w / 2, x + w / 2], [y - h / 2 + h * i / 4, y - h / 2 + h * i / 4], color="#A0A7AF", linewidth=0.45)
        axis.plot([x - w / 2 + w * i / 4, x - w / 2 + w * i / 4], [y - h / 2, y + h / 2], color="#A0A7AF", linewidth=0.45)
    for i, alpha in enumerate([0.85, 0.55, 0.30, 0.15]):
        axis.add_patch(Rectangle((x - w / 2 + 0.01 + i * w / 4, y + h / 2 - 0.04), w / 5, 0.025, facecolor="#4C78A8", edgecolor="none", alpha=alpha))


def _draw_teacher_icon(axis: plt.Axes, x: float, y: float, label: str, face: str, edge: str) -> None:
    axis.add_patch(Circle((x, y), 0.065, facecolor=face, edgecolor=edge, linewidth=1.1))
    for angle in [20, 100, 180, 260]:
        axis.add_patch(Arc((x, y), 0.12, 0.12, angle=angle, theta1=15, theta2=65, color=edge, linewidth=0.8))
    _label(axis, x, y - 0.12, label, size=6.7, weight="bold", box=True)


def _draw_artifact_icon(axis: plt.Axes, x: float, y: float) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x - 0.10, y - 0.07),
            0.20,
            0.14,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="#EAF3E8",
            edgecolor="#477A51",
            linewidth=1.1,
        )
    )
    for dy in [-0.035, 0.0, 0.035]:
        axis.plot([x - 0.065, x + 0.065], [y + dy, y + dy], color="#477A51", linewidth=0.9)
    _label(axis, x, y - 0.125, "response bank", size=6.4, weight="bold", color="#315C38", box=True)


def _draw_mechanism_band(axis: plt.Axes, xy: tuple[float, float], width: float, height: float) -> None:
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.024",
            facecolor="#FBFCFB",
            edgecolor="#D6DBD8",
            linewidth=0.8,
        )
    )


def _draw_spine_node(
    axis: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    label: str,
    edge: str,
    face: str,
    label_size: float = 6.3,
) -> None:
    x, y = center
    w, h = size
    axis.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.95,
        )
    )
    _label(axis, x, y, label, size=label_size, weight="bold", color=FIG1_COLORS["ink"])


def _plot_jpeg_aware_mechanism_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    colors = FIG1_COLORS
    axis.add_patch(
        FancyBboxPatch(
            (0.07, 0.08),
            0.86,
            0.82,
            boxstyle="round,pad=0.012,rounding_size=0.026",
            facecolor="#FBFCFB",
            edgecolor="#D6DBD8",
            linewidth=0.8,
        )
    )
    _label(axis, 0.50, 0.865, "same attack shell", size=6.6, weight="bold", color=colors["muted"])

    _label(axis, 0.24, 0.795, "shared state", size=6.8, weight="bold", color=colors["muted"])
    _draw_image_icon(axis, 0.24, 0.705, w=0.135, h=0.100)
    _label(axis, 0.24, 0.625, "clean image", size=6.1, weight="bold", box=True)
    _arrow(axis, (0.24, 0.590), (0.24, 0.525), colors["ink"], width=0.9)
    _draw_dct_icon(axis, 0.24, 0.455, w=0.145, h=0.112)
    _label(axis, 0.24, 0.370, "DCT coordinates", size=5.8, weight="bold", box=True)
    _arrow(axis, (0.24, 0.340), (0.24, 0.285), colors["ink"], width=0.9)
    _draw_spine_node(axis, (0.24, 0.220), (0.24, 0.070), "quality policy", colors["teacher"], "#F8E6C8", label_size=5.8)

    _label(axis, 0.67, 0.775, "calibrate", size=6.8, weight="bold", color=colors["teacher"])
    _arrow_path(axis, [(0.32, 0.455), (0.44, 0.455), (0.44, 0.655), (0.46, 0.655)], colors["muted"], width=0.9)
    _draw_spine_node(axis, (0.56, 0.655), (0.20, 0.070), "BPDA proxy", colors["proxy"], "#E8EEF5", label_size=5.7)
    _draw_spine_node(axis, (0.80, 0.655), (0.20, 0.070), "EOT teacher", colors["teacher"], "#FBE9D8", label_size=5.7)
    _arrow_path(axis, [(0.56, 0.620), (0.56, 0.560), (0.63, 0.560), (0.63, 0.520)], colors["response"], width=1.0)
    _arrow_path(axis, [(0.80, 0.620), (0.80, 0.560), (0.73, 0.560), (0.73, 0.520)], colors["response"], width=1.0)
    _draw_artifact_icon(axis, 0.68, 0.465)

    _label(axis, 0.54, 0.305, "reuse", size=6.3, weight="bold", color=colors["response"])
    _arrow_path(axis, [(0.36, 0.220), (0.45, 0.220), (0.45, 0.185), (0.46, 0.185)], colors["muted"], width=0.9)
    _arrow_path(axis, [(0.68, 0.340), (0.68, 0.265), (0.60, 0.265), (0.60, 0.225)], colors["response"], width=1.0)
    _draw_spine_node(axis, (0.60, 0.185), (0.28, 0.070), "CPU attack steps", colors["response"], "#EAF3E8", label_size=5.7)
    _arrow(axis, (0.74, 0.185), (0.775, 0.185), colors["response"], width=1.0)
    _draw_image_icon(axis, 0.82, 0.185, w=0.095, h=0.072)
    _label(axis, 0.76, 0.120, "compressed view", size=5.2, weight="bold", box=True)


def generate_plots(config: dict[str, object]) -> dict[str, str]:
    analysis = load_analysis(config)
    summary = analysis["summary"]
    tradeoff = analysis["efficiency_tradeoff"]
    plot_root = ensure_dir(resolve_project_path(config["paths"]["plot_root"]))
    manifest: dict[str, str] = {}

    static = summary[summary["suite"].isin(PRIMARY_STATIC_SUITES)].copy()
    if not static.empty:
        static["quality"] = static["suite"].map(_suite_quality)
        figure, axis = plt.subplots(figsize=(6, 4))
        for method, group in static.groupby("method"):
            ordered = group.sort_values("quality")
            axis.plot(ordered["quality"], ordered["asr_mean"], marker="o", label=method)
        axis.set_xlabel("JPEG quality")
        axis.set_ylabel("ASR")
        axis.set_title("Static-quality ASR")
        axis.legend()
        axis.grid(alpha=0.3)
        path = plot_root / "figure_static_trends.png"
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)
        manifest["figure_static_trends"] = str(path)

    if not tradeoff.empty:
        figure, axis = plt.subplots(figsize=(6.4, 4.6))
        _plot_tradeoff_axis(axis, tradeoff)
        path = plot_root / "figure_efficiency_tradeoff.png"
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)
        manifest["figure_efficiency_tradeoff"] = str(path)

        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        _plot_speedup_axis(axis, tradeoff)
        path = plot_root / "figure_speedup_bars.png"
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)
        manifest["figure_speedup_bars"] = str(path)

    mechanism_path = plot_root / "figure_jpeg_aware_mechanism.png"
    try:
        fig1_outputs = copy_cached_fig1_mermaid_assets(plot_root)
    except FileNotFoundError:
        fig1_outputs = render_fig1_mermaid_assets(plot_root)
    mechanism_path = fig1_outputs["figure_jpeg_aware_mechanism_png"]
    manifest["figure_jpeg_aware_mechanism"] = str(mechanism_path)
    manifest["fig1_mechanism"] = str(fig1_outputs["fig1_mechanism_pdf"])
    manifest["fig1_mechanism_svg"] = str(fig1_outputs["fig1_mechanism_svg"])

    if not tradeoff.empty:
        figure = _plot_cpu_advantage_structure_figure(tradeoff, _calibration_npz_for_config(config))
        structure_path = plot_root / "fig2_cpu_advantage_structure.pdf"
        figure.savefig(structure_path)
        _save_figure_svg(figure, plot_root / "fig2_cpu_advantage_structure.svg")
        plt.close(figure)
        manifest["fig2_cpu_advantage_structure"] = str(structure_path)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.axis("off")
    axis.text(
        0.05,
        0.9,
        "Appendix source-data summary\nDiffJPEG context, recovery-pass comparisons, and VF ablations are reported from source-data tables.",
        fontsize=11,
        va="top",
    )
    appendix_path = plot_root / "figure_appendix_summary.png"
    figure.tight_layout()
    figure.savefig(appendix_path, dpi=200)
    plt.close(figure)
    manifest["figure_appendix"] = str(appendix_path)

    manifest_path = plot_root / "plot_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def _load_reviewer_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _plot_empty_axis(axis: plt.Axes, message: str) -> None:
    axis.axis("off")
    axis.text(0.05, 0.9, message, fontsize=10, va="top")


def _plot_reviewer_matrix_axis(axis: plt.Axes, tradeoff: pd.DataFrame) -> None:
    if tradeoff.empty:
        _plot_empty_axis(axis, "No completed reviewer evidence source data are available.")
        return

    datasets = sorted(str(item) for item in tradeoff["dataset"].dropna().unique())
    color_map = {dataset: plt.get_cmap("tab10")(index % 10) for index, dataset in enumerate(datasets)}
    for dataset, group in tradeoff.groupby("dataset", sort=True):
        axis.scatter(
            group["runtime_speedup_mean"],
            group["asr_retention_mean"],
            s=60,
            label=str(dataset),
            color=color_map[str(dataset)],
            alpha=0.85,
        )
    for (dataset, model_family), group in tradeoff.groupby(["dataset", "model_family"], sort=True):
        label = f"{dataset} / {model_family}"
        axis.annotate(
            label,
            (group["runtime_speedup_mean"].median(), group["asr_retention_mean"].median()),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.8},
        )
    axis.axvline(1.0, color="#666666", linestyle="--", linewidth=0.9)
    axis.axhline(0.75, color="#666666", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Runtime speedup (baseline / VF)")
    axis.set_ylabel("ASR retention (VF / baseline)")
    axis.set_title("Reviewer-response evidence matrix")
    axis.legend()
    axis.grid(alpha=0.25)


def _plot_reviewer_ablation_axis(axis: plt.Axes, ablation: pd.DataFrame) -> None:
    if ablation.empty:
        _plot_empty_axis(axis, "No completed ablation or stability source data are available.")
        return

    labels = [f"{row.axis}: {row.setting}" for row in ablation.itertuples()]
    positions = range(len(ablation))
    axis.bar(positions, ablation["runtime_speedup_mean"], color="#4C78A8", alpha=0.8, label="Runtime speedup")
    axis.plot(positions, ablation["asr_retention_mean"], color="#F58518", marker="o", label="ASR retention")
    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9)
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_title("Ablation and stability source data")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)


def generate_reviewer_evidence_plots(evidence_root: str | Path) -> dict[str, str]:
    root = resolve_project_path(evidence_root)
    analysis_root = root / "analysis"
    plot_root = ensure_dir(root / "plots")
    tradeoff = _load_reviewer_csv(analysis_root / "reviewer_evidence_tradeoff.csv")
    ablation = _load_reviewer_csv(analysis_root / "reviewer_ablation_summary.csv")
    manifest: dict[str, str] = {}

    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    _plot_reviewer_matrix_axis(axis, tradeoff)
    matrix_path = plot_root / "figure_reviewer_evidence_matrix.png"
    figure.tight_layout()
    figure.savefig(matrix_path, dpi=200)
    plt.close(figure)
    manifest["figure_reviewer_evidence_matrix"] = str(matrix_path)

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    _plot_reviewer_ablation_axis(axis, ablation)
    ablation_path = plot_root / "figure_reviewer_ablation_stability.png"
    figure.tight_layout()
    figure.savefig(ablation_path, dpi=200)
    plt.close(figure)
    manifest["figure_reviewer_ablation_stability"] = str(ablation_path)

    manifest_path = plot_root / "reviewer_plot_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest
