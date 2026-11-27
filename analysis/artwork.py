"""Matplotlib artwork at final publication size; statistical inputs are unchanged."""
from itertools import combinations
import io

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

from .analysis_utils import FIGURE_DIR, MODELS, MODEL_LABELS

MM_PER_INCH = 25.4
WIDTH_MM = 174
RASTER_DPI = 600
FONT_PT = 8


def configure_artwork():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": FONT_PT,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .5,
        "mathtext.fontset": "dejavusans",
        "lines.linewidth": 1, "hatch.linewidth": .35,
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "#111111", "axes.labelcolor": "#111111",
        "xtick.color": "#111111", "ytick.color": "#111111",
    })


def export_artwork(fig, name, minimum_font_pt=8, raster_dpi=RASTER_DPI):
    """Keep the page size fixed; RGB raster alternatives accompany vector EPS/PDF."""
    width_mm, height_mm = fig.get_size_inches() * MM_PER_INCH
    if abs(width_mm-WIDTH_MM) > .01 or height_mm > 234:
        raise ValueError(f"Invalid final size for {name}: {width_mm} x {height_mm} mm")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    # Matplotlib retains tick-label artists outside the view interval but does
    # not draw them; only inspect labels that are actually part of the page.
    hidden_ticks = set()
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    hidden_ticks.update([tick.label1, tick.label2])
    for artist in fig.findobj(match=Text):
        if artist in hidden_ticks or not artist.get_visible() or not artist.get_text():
            continue
        if not minimum_font_pt <= artist.get_fontsize() <= 12:
            raise ValueError(f"Lettering outside the selected size range: {artist.get_text()}")
        box = artist.get_window_extent(renderer)
        if box.x0 < page.x0-.5 or box.y0 < page.y0-.5 or box.x1 > page.x1+.5 or box.y1 > page.y1+.5:
            raise ValueError(f"Text outside page in {name}: {artist.get_text()}")
    FIGURE_DIR.mkdir(exist_ok=True)
    for suffix in ["eps", "pdf"]:
        fig.savefig(FIGURE_DIR/f"{name}.{suffix}", facecolor="white", transparent=False)
    # Convert the canvas to 8-bit RGB rather than leaving an alpha channel.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=raster_dpi, facecolor="white", transparent=False)
    buffer.seek(0)
    with Image.open(buffer) as source:
        rgb = source.convert("RGB")
        # Finish the PNG before replacing the previewed file on Windows.
        png_path = FIGURE_DIR/f"{name}.png"
        temporary_png = FIGURE_DIR/f".{name}.tmp.png"
        rgb.save(temporary_png, dpi=(raster_dpi, raster_dpi))
        temporary_png.replace(png_path)
        rgb.save(FIGURE_DIR/f"{name}.tif", compression="tiff_lzw", dpi=(raster_dpi, raster_dpi))
    plt.close(fig)
    print(f"Saved {name}: {width_mm:.0f} x {height_mm:.1f} mm, EPS/PDF and {raster_dpi}-dpi RGB PNG/TIFF")


def save_original_artwork(fig, name):
    """Fixed 174 x 234 mm page; retain the original artists, fonts, and layout."""
    width_mm, height_mm = fig.get_size_inches() * MM_PER_INCH
    if not np.allclose([width_mm, height_mm], [174, 234], rtol=0, atol=.01):
        raise ValueError(f"Unexpected page dimensions: {width_mm} x {height_mm}")
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.canvas.draw()
    for suffix in ["pdf", "eps"]:
        fig.savefig(FIGURE_DIR/f"{name}.{suffix}", bbox_inches=None, dpi=RASTER_DPI)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches=None, dpi=RASTER_DPI)
    buffer.seek(0)
    with Image.open(buffer) as image:
        rgb = image.convert("RGB")
        target = FIGURE_DIR/f"{name}.png"
        temporary = FIGURE_DIR/f".{name}.tmp.png"
        rgb.save(temporary, dpi=(RASTER_DPI, RASTER_DPI))
        temporary.replace(target)
        rgb.save(FIGURE_DIR/f"{name}.tif", compression="tiff_lzw", dpi=(RASTER_DPI, RASTER_DPI))
    plt.close(fig)
    print(f"Saved {name}: 174 x 234 mm, original layout, 600-dpi PNG")


def plot_timing_sensitivity(sweep):
    """Plot CNEP+ matched-cohort counts and MAD for each follow-up tolerance."""
    if set(sweep.Model) != set(MODELS):
        raise ValueError("Timing sensitivity requires the three evaluated models.")
    if sweep.duplicated(["Model", "tolerance_years"]).any():
        raise ValueError("Duplicate model/tolerance records in timing sensitivity.")
    if not np.isfinite(sweep[["tolerance_years", "n_agents", "MAD"]].to_numpy(float)).all():
        raise ValueError("Timing sensitivity requires finite tolerances, cohort counts, and MADs.")
    if sweep.tolerance_years.le(0).any() or sweep.n_agents.le(0).any():
        raise ValueError("Timing tolerances and cohort counts must be positive.")
    configure_artwork()
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_MM/MM_PER_INCH, 130/MM_PER_INCH), sharex=True)
    fig.subplots_adjust(left=.16, right=.97, bottom=.12, top=.97, hspace=.12)
    counts = sweep.pivot(index="tolerance_years", columns="Model", values="n_agents").sort_index(ascending=False)
    if counts.isna().any().any() or not counts.eq(counts.iloc[:,0], axis=0).all().all():
        raise ValueError("Timing sensitivity has different cohort counts across models.")
    axes[0].plot(counts.index, counts.iloc[:,0], color="#111111", marker="o", markersize=4, linewidth=1)
    colors, markers, lines = ["#245A81", "#A33F25", "#267142"], ["o", "s", "^"], ["-", "--", ":"]
    for model, color, marker, line in zip(MODELS, colors, markers, lines):
        rows = sweep.loc[sweep.Model.eq(model)].sort_values("tolerance_years", ascending=False)
        axes[1].plot(rows.tolerance_years, rows.MAD, color=color, marker=marker, linestyle=line,
                     markersize=4, linewidth=1.1, label=MODEL_LABELS[model])
    axes[0].set_ylabel("Agents retained")
    axes[1].set_ylabel("CNEP+ correlation MAD")
    axes[1].set_xlabel("Tolerance around six-month target (± years)")
    axes[1].set_xscale("log"); axes[1].invert_xaxis()
    axes[1].set_xticks(counts.index, [f"{v:g}" for v in counts.index])
    axes[1].legend(loc="upper left", bbox_to_anchor=(.08, 1), frameon=False)
    for ax, letter in zip(axes, "ab"):
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(.06, ax.get_position().y1, f"({letter})", ha="left", va="top", fontsize=9, weight="bold")
    # Springer specifies at least 1200 dpi for bitmap line drawings.
    export_artwork(fig, "FigS3", raster_dpi=1200)


def plot_cnep_correlations(results_df):
    """Compare housing-specific LLM correlations with matched CNEP+ follow-up."""
    import string
    with plt.rc_context(plt.rcParamsDefault):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        plot_df = results_df.copy()
        plot_df["Homeless"] = plot_df["Housing"].map({0: "Stable", 1: "Homeless"})
        plot_df["Model"] = plot_df["Model"].replace({
            "claudesonnet4.6": "Claude",
            "us.anthropic.claude-sonnet-4-6": "Claude",
            "deepseek.v3.2": "DeepSeek",
            "gpt-oss-120b": "GPT-OSS",
            "openai.gpt-oss-120b-1:0": "GPT-OSS",
        })

        model_order = ["Claude", "DeepSeek", "GPT-OSS"]
        homeless_order = ["Stable", "Homeless"]

        if not plot_df["Model"].isin(model_order).all():
            raise ValueError("Add unknown model names to plot display mapping.")
        if (plot_df["Homeless"].isna().any()
                or plot_df.duplicated(["Model", "Homeless"]).any()
                or set(zip(plot_df.Model, plot_df.Homeless))
                != {(model, housing) for model in model_order for housing in homeless_order}):
            raise ValueError("CNEP+ comparison requires one row per model and housing condition.")

        plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_order, ordered=True)
        plot_df["Homeless"] = pd.Categorical(plot_df["Homeless"], categories=homeless_order, ordered=True)

        plot_df["Label"] = plot_df["Model"].astype(str) + " – " + plot_df["Homeless"].astype(str)
        order = plot_df.sort_values(["Model", "Homeless"])["Label"].drop_duplicates().tolist()
        plot_df["Label"] = pd.Categorical(plot_df["Label"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("Label").reset_index(drop=True)

        model_colors = {"Claude": "#4C72B0", "DeepSeek": "#DD8452", "GPT-OSS": "#55A868"}
        housing_markers = {"Stable": "o", "Homeless": "^"}

        compare_cols = ["Injections", "Recept_sharing", "Drug_in", "Drug_out", "Partner_HCV_prop"]
        pretty_names = {
            "Injections": "Injections",
            "Recept_sharing": "Receptive Sharing",
            "Drug_in": "Network In-Degree",
            "Drug_out": "Network Out-Degree",
            "Partner_HCV_prop": "Partners with HCV",}

        metrics = []
        for x, y in combinations(compare_cols, 2):
            llm_col = f"assoc_LLM_{x}__{y}"
            cnep_col = f"assoc_CNEP_{x}__{y}"
            title = f"{pretty_names[x]} ~ {pretty_names[y]}"
            metrics.append((llm_col, cnep_col, title))

        n_rows = 5
        n_cols = 2
        fig_width = 174 / 25.4
        fig_height = 234 / 25.4

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=False)
        axes = axes.flatten()

        y = np.arange(len(plot_df))
        subfig_labels = list(string.ascii_lowercase)

        for i, (ax, (llm_col, cnep_col, title)) in enumerate(zip(axes, metrics)):
            ax.scatter(plot_df[cnep_col], y, color="black", marker="|", s=100, zorder=2, label="Matched CNEP+")

            for model, sub in plot_df.groupby("Model", observed=True):
                for housing, sub2 in sub.groupby("Homeless", observed=True):
                    idx = sub2.index
                    ax.scatter(plot_df.loc[idx, llm_col], y[idx], color=model_colors[model],
                        marker=housing_markers[housing], s=55, edgecolor="white", linewidth=0.5, zorder=3,)

            ax.set_yticks(y)
            ax.set_yticklabels(plot_df["Label"], fontsize=7)
            ax.set_ylim(-0.5, len(plot_df) - 0.5)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=8, loc="left")
            ax.set_xlim(-1, 1)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.text(0.5, -0.32, f"({subfig_labels[i]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, fontname="Arial",)

        model_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=7, label=m)
            for m, c in model_colors.items()]
        housing_handles = [plt.Line2D([0], [0], marker=mk, color="gray", linestyle="", markersize=7, label=h)
            for h, mk in housing_markers.items()]
        cnep_handle = [plt.Line2D([0], [0], color="black", marker="|", linestyle="", label="Matched CNEP+")]

        fig.supxlabel("Spearman’s rank correlation", fontsize=8, y=0.045)

        fig.legend(handles=model_handles + housing_handles + cnep_handle, loc="lower center",
            ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.005),)

        plt.tight_layout(rect=[0, 0, 1, 1])

        save_original_artwork(fig, "FigS2")
