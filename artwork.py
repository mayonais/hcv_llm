"""Matplotlib artwork at final publication size; statistical inputs are unchanged."""
from itertools import combinations
import io
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

from analysis_utils import COLUMNS, EXPECTED, FIGURE_DIR, MODELS, TERMS

MM_PER_INCH = 25.4
WIDTH_MM = 174
RASTER_DPI = 600
FONT_PT = 8
STYLES = {
    "match": ("#D8EAD4", ""),
    "opposite": ("#F0CBCB", "xx"),
    "undefined": ("#F4E2BC", ".."),
    "unscored": ("#E3E6E8", "///"),
    "uncertain": ("#DBE6EF", "\\\\"),
}
NAMES = {
    "Homeless": "Homelessness", "Public_prop": "Public injecting",
    "Overdose": "Non-fatal overdose", "Recept_sharing": "Receptive sharing",
    "Injections": "Injection frequency", "Treatment": "Treatment",
    "HCV_active": "Positive HCV status", "Alt_route": "Non-injection routes",
    "Partners": "Injection partners", "Partner_HCV_prop": "Partners with HCV",
    "Drug_in": "Network in-degree", "Drug_out": "Network out-degree",
}
SHORT_MODELS = dict(zip(MODELS, ["Claude", "DeepSeek", "GPT-OSS"]))


def configure_artwork():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": FONT_PT,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "axes.linewidth": .5,
        "mathtext.fontset": "dejavusans",
        "lines.linewidth": 1, "hatch.linewidth": .35,
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "#111111", "axes.labelcolor": "#111111",
        "xtick.color": "#111111", "ytick.color": "#111111",
    })


def export_artwork(fig, name):
    """Keep the page size fixed; RGB raster alternatives accompany vector EPS/PDF."""
    width_mm, height_mm = fig.get_size_inches() * MM_PER_INCH
    if abs(width_mm-WIDTH_MM) > .01 or height_mm > 234:
        raise ValueError(f"Invalid final size for {name}: {width_mm} x {height_mm} mm")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    fonts = []
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
        fonts.append(artist.get_fontsize())
        if not 8 <= artist.get_fontsize() <= 12:
            raise ValueError(f"Lettering outside 8--12 pt: {artist.get_text()}")
        box = artist.get_window_extent(renderer)
        if box.x0 < page.x0-.5 or box.y0 < page.y0-.5 or box.x1 > page.x1+.5 or box.y1 > page.y1+.5:
            raise ValueError(f"Text outside page in {name}: {artist.get_text()}")
    FIGURE_DIR.mkdir(exist_ok=True)
    for suffix in ["eps", "pdf", "svg"]:
        fig.savefig(FIGURE_DIR/f"{name}.{suffix}", facecolor="white", transparent=False)
    # Convert the canvas to 8-bit RGB rather than leaving an alpha channel.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=RASTER_DPI, facecolor="white", transparent=False)
    buffer.seek(0)
    with Image.open(buffer) as source:
        rgb = source.convert("RGB")
        rgb.save(FIGURE_DIR/f"{name}.png", dpi=(RASTER_DPI, RASTER_DPI))
        rgb.save(FIGURE_DIR/f"{name}.tif", compression="tiff_lzw", dpi=(RASTER_DPI, RASTER_DPI))
    info = {"figure": name, "width_mm": float(width_mm), "height_mm": float(height_mm),
            "minimum_font_pt": min(fonts), "maximum_font_pt": max(fonts),
            "raster_dpi": RASTER_DPI, "raster_mode": "RGB", "graphics_program": "Matplotlib"}
    audit_path = FIGURE_DIR/"artwork_checks.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    audit[name] = info
    audit_path.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    plt.close(fig)
    print(f"Saved {name}: {width_mm:.0f} x {height_mm:.1f} mm, EPS/PDF/SVG and 600-dpi RGB PNG/TIFF")


def _arrow(value):
    return "↑" if value > 0 else "↓" if value < 0 else "—"


def _status(expected, direction=None, value=None):
    if value is not None and not np.isfinite(value):
        return "undefined"
    if expected == "↔":
        return "unscored"
    if direction in ["mixed", "uncertain"]:
        return "uncertain"
    actual = {"increase": "↑", "decrease": "↓"}.get(direction) if direction else _arrow(value)
    return "match" if actual == expected else "opposite"


def _knowledge_text(row):
    direction = {"increase": "↑", "decrease": "↓", "mixed": "↔", "uncertain": "—"}[row.direction]
    lines = [f"{direction} {row.evidence_quality}"]
    if pd.notna(row.value):
        measure = "" if pd.isna(row.measure) else str(row.measure)
        lines.append(f"{measure} {row.value:.2f}".strip())
        if pd.notna(row.lower_ci) and pd.notna(row.upper_ci):
            lines.append(f"({row.lower_ci:.2f}–{row.upper_ci:.2f})")
    return "\n".join(lines)


def draw_association_table(kind, knowledge, housing, correlations):
    configure_artwork()
    pairs = [pair for pair in EXPECTED if (pair[0] == TERMS["Homeless"]) == (kind == "housing")]
    # Physical mm coordinates make font sizes independent of the number of rows.
    row_mm, header_mm, legend_mm, margin_mm = 8.6, 17.5, 10, 2
    height = 2*margin_mm + header_mm + row_mm*len(pairs) + legend_mm
    fig = plt.figure(figsize=(WIDTH_MM/MM_PER_INCH, height/MM_PER_INCH), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM); ax.set_ylim(0, height); ax.axis("off")
    left, label_w, expected_w, cell_w = 3, 40, 8, 20
    knowledge_x = left+label_w+expected_w
    output_x = knowledge_x+3*cell_w
    group_y = height-margin_mm-2
    ax.text(knowledge_x+1.5*cell_w, group_y, "LLM knowledge probe", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(output_x+1.5*cell_w, group_y, "LLM-generated outputs", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(left+1, height-10, "Variable pair", va="center", fontsize=8, weight="bold")
    ax.text(left+label_w+expected_w/2, height-10, "Exp.\ndir.", ha="center", va="center", fontsize=8)
    headers = ["Claude\nSonnet 4.6", "DeepSeek\nv3.2", "GPT-OSS\n120B"]
    for origin in [knowledge_x, output_x]:
        for j, label in enumerate(headers):
            ax.text(origin+(j+.5)*cell_w, height-9, label, ha="center", va="center", fontsize=8, linespacing=1)
    ax.text(knowledge_x+1.5*cell_w, height-15, "Direction, evidence, estimate", ha="center", va="center", fontsize=8)
    generated_label = "Paired $d_z$ (95% bootstrap CI)" if kind == "housing" else "Spearman rank correlation"
    ax.text(output_x+1.5*cell_w, height-15, generated_label, ha="center", va="center", fontsize=8)
    body_top = height-margin_mm-header_mm
    cell_records = []

    def cell(x, y, text, status):
        face, hatch = STYLES[status]
        ax.add_patch(Rectangle((x+.3, y+.2), cell_w-.6, row_mm-.4,
                               facecolor=face, edgecolor="#444444", linewidth=.35, hatch=hatch))
        artist = ax.text(x+cell_w/2, y+row_mm/2, text, ha="center", va="center", fontsize=8,
                        linespacing=.95, color="#111111",
                        bbox={"facecolor": face, "edgecolor": "none", "pad": .15})
        cell_records.append((artist, cell_w-.8, row_mm-.4))

    for index, (exposure, outcome) in enumerate(pairs):
        y = body_top-(index+1)*row_mm
        a, b = COLUMNS[exposure], COLUMNS[outcome]
        expected = EXPECTED[(exposure, outcome)]
        ax.text(left+1, y+row_mm/2, f"{NAMES[a]}\n~ {NAMES[b]}", ha="left", va="center", fontsize=8, linespacing=1)
        ax.text(left+label_w+expected_w/2, y+row_mm/2, expected, ha="center", va="center", fontsize=10)
        for j, model in enumerate(MODELS):
            rows = knowledge.loc[knowledge.Model.eq(model)&knowledge.exposure.eq(exposure)&knowledge.outcome.eq(outcome)]
            if len(rows) != 1:
                raise ValueError(f"Expected one knowledge record: {model}/{exposure}/{outcome}")
            row = rows.iloc[0]
            cell(knowledge_x+j*cell_w, y, _knowledge_text(row), _status(expected, direction=row.direction))
            if kind == "housing":
                row = housing.loc[housing.Model.eq(model)&housing.Variable.eq(b)&housing.Paraphrase.eq(0)].iloc[0]
                value = row.cohens_d
                text = "—" if not np.isfinite(value) else f"{_arrow(value)} {value:.2f}"
                if np.isfinite(value):
                    if np.isfinite(row.d_ci_lower) and np.isfinite(row.d_ci_upper):
                        text += f"\n({row.d_ci_lower:.2f}–{row.d_ci_upper:.2f})"
                    else:
                        text += "\nCI n.e."
            else:
                row = correlations.loc[correlations.model.eq(model)&
                    ((correlations.var1.eq(a)&correlations.var2.eq(b))|(correlations.var1.eq(b)&correlations.var2.eq(a)))].iloc[0]
                value = row.correlation
                text = "—" if not np.isfinite(value) else f"{_arrow(value)} {value:.2f}"
            cell(output_x+j*cell_w, y, text, _status(expected, value=value))
    legend = [("match", "Matches expected direction"), ("opposite", "Opposite direction"),
              ("undefined", "Non-estimable"), ("unscored", "No fixed benchmark"),
              ("uncertain", "Mixed/uncertain answer")]
    for index, (status, label) in enumerate(legend):
        x = left+(index%3)*56
        y = 7.7-(index//3)*4.5
        face, hatch = STYLES[status]
        ax.add_patch(Rectangle((x, y-1.3), 4, 2.6, facecolor=face, edgecolor="#444444", linewidth=.35, hatch=hatch))
        ax.text(x+5.5, y, label, va="center", fontsize=8)
    # Check the text physically fits its cell before any export.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels_per_mm = fig.dpi/MM_PER_INCH
    for artist, width, row_height in cell_records:
        bounds = artist.get_window_extent(renderer)
        if bounds.width > width*pixels_per_mm or bounds.height > row_height*pixels_per_mm:
            raise ValueError(f"Text does not fit a publication cell: {artist.get_text()}")
    export_artwork(fig, "Fig2" if kind == "housing" else "FigS1")


def plot_timing_sensitivity(sweep):
    configure_artwork()
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_MM/MM_PER_INCH, 130/MM_PER_INCH), sharex=True)
    fig.subplots_adjust(left=.16, right=.97, bottom=.12, top=.97, hspace=.12)
    counts = sweep.pivot(index="tolerance_years", columns="Model", values="n_agents").sort_index(ascending=False)
    if not counts.eq(counts.iloc[:,0], axis=0).all().all():
        raise ValueError("Timing sensitivity has different cohort counts across models.")
    axes[0].plot(counts.index, counts.iloc[:,0], color="#111111", marker="o", markersize=4, linewidth=1)
    colors, markers, lines = ["#245A81", "#A33F25", "#267142"], ["o", "s", "^"], ["-", "--", ":"]
    for model, color, marker, line in zip(MODELS, colors, markers, lines):
        rows = sweep.loc[sweep.Model.eq(model)].sort_values("tolerance_years", ascending=False)
        axes[1].plot(rows.tolerance_years, rows.MAD, color=color, marker=marker, linestyle=line,
                     markersize=4, linewidth=1.1, label=SHORT_MODELS[model])
    axes[0].set_ylabel("Agents retained")
    axes[1].set_ylabel("ABM correlation MAD")
    axes[1].set_xlabel("Tolerance around six-month target (± years)")
    axes[1].set_xscale("log"); axes[1].invert_xaxis()
    axes[1].set_xticks(counts.index, [f"{v:g}" for v in counts.index])
    axes[1].legend(loc="upper left", bbox_to_anchor=(.08, 1), frameon=False)
    for ax, letter in zip(axes, "ab"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.015, .95, letter, transform=ax.transAxes, va="top", fontsize=9, weight="bold")
    export_artwork(fig, "FigS3")


def plot_abm_correlations(results):
    configure_artwork()
    frame = results.copy()
    model_map = {**SHORT_MODELS, "us.anthropic.claude-sonnet-4-6": "Claude", "openai.gpt-oss-120b-1:0": "GPT-OSS"}
    frame["ModelLabel"] = frame.Model.map(model_map)
    frame["HousingLabel"] = frame.Housing.map({0:"Stable", 1:"Homeless"})
    if frame[["ModelLabel", "HousingLabel"]].isna().any().any():
        raise ValueError("Unknown model or housing condition in ABM plot.")
    frame["ModelLabel"] = pd.Categorical(frame.ModelLabel, ["Claude", "DeepSeek", "GPT-OSS"], ordered=True)
    frame = frame.sort_values(["ModelLabel", "Housing"]).reset_index(drop=True)
    labels = frame.ModelLabel.astype(str)+" – "+frame.HousingLabel
    cols = ["Injections", "Recept_sharing", "Drug_in", "Drug_out", "Partner_HCV_prop"]
    fig, axes = plt.subplots(5, 2, figsize=(WIDTH_MM/MM_PER_INCH, 220/MM_PER_INCH))
    fig.subplots_adjust(left=.18, right=.98, top=.975, bottom=.115, hspace=1.1, wspace=.92)
    colors = {"Claude":"#245A81", "DeepSeek":"#A33F25", "GPT-OSS":"#267142"}
    for letter, ax, (a,b) in zip("abcdefghij", axes.flat, combinations(cols,2)):
        y = np.arange(len(frame))
        ax.scatter(frame[f"assoc_ABM_{a}__{b}"], y, marker="|", s=100, color="#111111", linewidths=1, zorder=2)
        for i, row in frame.iterrows():
            ax.scatter(row[f"assoc_LLM_{a}__{b}"], i, marker="o" if row.Housing==0 else "^", s=25,
                       color=colors[row.ModelLabel], edgecolors="#111111", linewidths=.4, zorder=3)
        ax.set_yticks(y, labels, fontsize=8); ax.set_ylim(-.5,len(frame)-.5); ax.invert_yaxis()
        ax.set_xlim(-1,1); ax.set_xticks([-1,0,1])
        ax.set_xlabel(f"{NAMES[a]}\n~ {NAMES[b]}", fontsize=8, labelpad=2)
        ax.axvline(0, linewidth=.35, color="#555555", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.03,.92,letter,transform=ax.transAxes,va="top",fontsize=9,weight="bold")
    fig.supxlabel("Spearman rank correlation", fontsize=9, y=.045)
    handles = [plt.Line2D([],[],marker=mark,linestyle="",color="#111111",label=label,markersize=5)
               for mark,label in [("o","Stable"),("^","Homeless"),("|","Matched ABM")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.56,.005), fontsize=8)
    export_artwork(fig, "FigS2")
