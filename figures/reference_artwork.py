"""Editable vector reconstruction of the supplied Fig2 and FigS1 artwork.

Values come from the current llm_outputs and analysis_results CSVs. Coordinates are PDF
points measured from the references; there is no cropped-page or image replay.
"""
from pathlib import Path
import sys

import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analysis_utils import (
    COLUMNS, HOUSING_UNITS, MODELS, MODEL_LABELS, expected_directions,
    normalize_knowledge, housing_figure_values, correlation_figure_values,
)

BODY_PT = 8.2
STABILITY_PT = BODY_PT - 1
FIG2_EXTRA_PANEL_GAP_PT = 8
COLORS = {
    "text": "#1A1A1A", "header": "#B0B8C4", "stripe": "#F0F4F8",
    "match": "#8BCB87", "opposite": "#E57373", "null": "#FBDCA3",
    "mixed": "#ECEFF1", "hatch": "#D0D7DB", "uncertain": "#A8D8A4",
    "zero": "#E3E6E8",
}
NAMES = {
    "Homeless": "Homelessness", "Public_prop": "Public injecting",
    "Overdose": "Non-fatal overdose", "Recept_sharing": "Receptive sharing",
    "Injections": "Injection frequency", "Treatment": "Treatment",
    "HCV_active": "Positive HCV status", "Alt_route": "Non-injection routes",
    "Partners": "Injection partners", "Partner_HCV_prop": "Partners with HCV",
}
DIRECTION = {"increase": "↑", "decrease": "↓", "mixed": "↔", "uncertain": "—"}


def register_fonts():
    """Bundle the same Liberation Sans faces embedded by the reference PDFs."""
    for style, suffix in [("regular", "Regular"), ("bold", "Bold"), ("italic", "Italic")]:
        name = f"ReferenceSans-{style}"
        if name not in pdfmetrics.getRegisteredFontNames():
            path = ROOT / "figures" / "assets" / "fonts" / f"LiberationSans-{suffix}.ttf"
            pdfmetrics.registerFont(TTFont(name, str(path)))


def _font(style="regular"):
    return f"ReferenceSans-{style}"


def _width(text, size=BODY_PT, style="regular"):
    return pdfmetrics.stringWidth(str(text), _font(style), size)


def _text(c, x, y, text, size=BODY_PT, style="regular", centered=False):
    # The source uses exactly 0.1 rather than the rounded 8-bit value 26/255.
    c.setFillColorRGB(.1, .1, .1)
    c.setFont(_font(style), size)
    if centered:
        x -= _width(text, size, style) / 2
    c.drawString(x, y, str(text))


def _center_lines(c, cx, cy, lines, size=BODY_PT, style="regular", centered=True):
    baseline = cy - 3.5 + (len(lines) - 1) * 4.7
    for line in lines:
        _text(c, cx, baseline, line, size, style, centered)
        baseline -= 9.4


def _direction_line(c, cx, baseline, text, arrow_pt):
    """Keep each arrow + label group centered while sizing its arrow separately."""
    arrow, separator, rest = text.partition(" ")
    if arrow not in {"↑", "↓", "↔", "—"}:
        _text(c, cx, baseline, text, centered=True)
        return
    gap = _width(" ") if separator else 0
    total = _width(arrow, arrow_pt) + gap + _width(rest)
    left = cx - total / 2
    _text(c, left, baseline - .30 * (arrow_pt - BODY_PT), arrow, arrow_pt)
    if rest:
        _text(c, left + _width(arrow, arrow_pt) + gap, baseline, rest)


def _status(expected, direction, finite=True):
    if not finite:
        return "null"
    if expected == "↔":
        return "mixed"
    if direction in {"mixed", "uncertain"}:
        return "uncertain"
    if direction == "—":
        return "zero"
    return "match" if DIRECTION.get(direction, direction) == expected else "opposite"


def load_rows(data_dir=ROOT / "llm_outputs", result_dir=ROOT / "analysis_results"):
    """Read current responses and analysis results for every displayed cell."""
    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    required = [data_dir / "knowledge_outputs.csv", result_dir / "housing_effects.csv",
                result_dir / "pooled_outcome_correlations.csv", result_dir / "bootstrap_summary.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing figure inputs: " + ", ".join(missing) + ". "
            "Ensure llm_outputs/knowledge_outputs.csv exists and run "
            "analysis/llm_outputs_analysis.ipynb, then analysis/bootstrap.ipynb to generate the results."
        )
    knowledge = normalize_knowledge(pd.read_csv(data_dir / "knowledge_outputs.csv"))
    knowledge = knowledge.loc[knowledge.Paraphrase.eq(0)].set_index(["exposure", "outcome", "Model"])
    housing = pd.read_csv(result_dir / "housing_effects.csv")
    correlations = pd.read_csv(result_dir / "pooled_outcome_correlations.csv")
    bootstrap = pd.read_csv(result_dir / "bootstrap_summary.csv")
    generated = {"housing": housing_figure_values(housing, bootstrap),
                 "correlation": correlation_figure_values(correlations, bootstrap)}
    result = {}
    for kind in ["housing", "correlation"]:
        rows = []
        for pair, expected in expected_directions(kind).items():
            a, b = [NAMES[COLUMNS[term]] for term in pair]
            row = {"pair": pair, "label": f"{a} ~ {b[0].lower() + b[1:]}",
                   "expected": expected, "knowledge": [], "generated": []}
            if kind == "housing":
                unit = HOUSING_UNITS[COLUMNS[pair[1]]]
                row["generated_label"] = (f"{b} (percentage points)" if unit == "pp" else f"{b} ({unit})")
            for index, model in enumerate(MODELS):
                k = knowledge.loc[(*pair, model)]
                lines = [f"{DIRECTION[k.direction]} {k.evidence_quality}"]
                if pd.notna(k.value):
                    measure = "" if pd.isna(k.measure) else str(k.measure) + " "
                    estimate = f"{measure}{k.value:.2f}"
                    if pd.notna(k.lower_ci) and pd.notna(k.upper_ci):
                        estimate += f" ({k.lower_ci:.2f}–{k.upper_ci:.2f})"
                    lines.append(estimate)
                row["knowledge"].append({"lines": lines, "status": _status(expected, k.direction)})
                value = generated[kind][pair][index]
                finite = value.splitlines()[0] != "NaN"
                text = value if finite else value.replace("NaN", "—", 1)
                direction = text.split()[0]
                row["generated"].append({"lines": text.splitlines(),
                    "status": _status(expected, direction, finite)})
            rows.append(row)
        result[kind] = rows
    return result


def _cell(c, x, bottom, pitch, row_height, cell, arrow_pt, estimate_pt):
    width, height = pitch - 1, row_height - 1
    c.setFillColor(HexColor(COLORS[cell["status"]]))
    c.rect(x + .5, bottom + .5, width, height, fill=1, stroke=0)
    # Reference hatches and crosses use a slightly inset clipping rectangle.
    inner_x = x + .5 + width * 2 / 79
    inner_y = bottom + .5 + height * .0257734
    inner_w = width * 75 / 79
    inner_h = height * .9484532
    if cell["status"] == "opposite":
        c.saveState()
        c.setStrokeColorRGB(1, 1, 1)
        c.setLineWidth(.9)
        c.line(inner_x, inner_y, inner_x + inner_w, inner_y + inner_h)
        c.line(inner_x, inner_y + inner_h, inner_x + inner_w, inner_y)
        c.restoreState()
    elif cell["status"] == "mixed":
        c.saveState()
        clipping = c.beginPath()
        clipping.rect(inner_x, inner_y, inner_w, inner_h)
        c.clipPath(clipping, stroke=0, fill=0)
        c.setStrokeColor(HexColor(COLORS["hatch"]))
        c.setLineWidth(.6)
        for i in range(15):
            end_x = inner_x + i * inner_w * .14666667
            c.line(end_x - inner_w * .98133333, inner_y, end_x, inner_y + inner_h)
        c.restoreState()
    cx, cy = x + pitch / 2, bottom + row_height / 2
    baseline = cy - 3.5 + (len(cell["lines"]) - 1) * 4.7
    _direction_line(c, cx, baseline, cell["lines"][0], arrow_pt)
    for line in cell["lines"][1:]:
        baseline -= 9.4
        line_pt = STABILITY_PT if line.startswith("agreement ") else estimate_pt
        if line.startswith("agreement ") and _width(line, line_pt) > width - 2:
            raise ValueError(f"Bootstrap agreement exceeds the model-cell width: {line}")
        _text(c, cx, baseline, line, line_pt, centered=True)


def _panel(c, width, height, offset, rows, kind, knowledge, arrow_pt,
           expected_arrow_pt, knowledge_estimate_pt):
    housing = kind == "housing"
    title = "(a) LLM Knowledge Probe" if knowledge else (
        "(b) LLM-Generated Behavioral Outputs" if housing else "(b) LLM-Generated Outputs")
    _text(c, 0, height - offset - 10, title, 10, "bold")
    if knowledge:
        subtitle = "LLM-reported association direction, evidence strength, and association estimate"
        _text(c, 3, height - offset - 22.5, subtitle, style="italic")
    elif housing:
        _text(c, 3, height - offset - 22.5,
              "Mean paired difference: homeless minus stable; bootstrap directional agreement", style="italic")
    else:
        _text(c, 3, height - offset - 22.5,
              "Spearman's rank correlation; bootstrap directional agreement", style="italic")
    header_bottom = height - offset - 55
    c.setFillColor(HexColor(COLORS["header"]))
    c.rect(0, header_bottom, width, 30, fill=1, stroke=0)
    _text(c, 3, header_bottom + 11.5,
          "Outcome (unit)" if housing and not knowledge else "Variable pairs", style="bold")
    model_x = 208 if housing else 210
    expected_x = 180 if housing else 184
    pitch = (width - model_x) / 3
    expected_lines = ["Literature", "Direction"]
    _center_lines(c, expected_x, header_bottom + 15, expected_lines, style="bold")
    for i, model in enumerate(MODELS):
        _text(c, model_x + (i + .5) * pitch, header_bottom + 11.5,
              MODEL_LABELS[model], style="bold", centered=True)
    row_height = 22.8 if housing else 21
    for i, row in enumerate(rows):
        bottom = header_bottom - (i + 1) * row_height
        if i % 2 == 0:
            c.setFillColor(HexColor(COLORS["stripe"]))
            c.rect(0, bottom, width, row_height, fill=1, stroke=0)
        cy = bottom + row_height / 2
        label = row["generated_label"] if housing and not knowledge else row["label"]
        direction_width = max(_width(line, style="bold") for line in expected_lines)
        if 3 + _width(label) + 8 > expected_x - direction_width / 2:
            raise ValueError(f"Single-line label overlaps the direction column: {label}")
        _text(c, 3, cy - 3.5, label)
        _text(c, expected_x, cy - 5.3 - .30 * (expected_arrow_pt - 10),
              row["expected"], expected_arrow_pt, "bold", centered=True)
        for j, cell in enumerate(row["knowledge" if knowledge else "generated"]):
            _cell(c, model_x + j * pitch, bottom, pitch, row_height, cell,
                  arrow_pt, knowledge_estimate_pt if knowledge else BODY_PT)


def _legend(c, width, supplementary):
    separator = 45
    c.setStrokeColorRGB(.65, .65, .65)
    c.setLineWidth(.4)
    c.line(0, separator, width, separator)
    entries = [("match", "Matches literature direction"), ("opposite", "Opposite of literature direction"),
               ("null", "Undefined correlation") if supplementary else ("zero", "Zero mean difference"),
               ("mixed", "Literature direction is mixed")]
    if supplementary:
        entries.append(("uncertain", "Mixed or uncertain knowledge direction"))
    for i, (status, label) in enumerate(entries):
        x = (i % 2) * width / 2
        y = separator - 14 - (i // 2) * 12.8
        c.setFillColor(HexColor(COLORS[status]))
        c.rect(x, y, 15, 7, fill=1, stroke=0)
        if status == "opposite":
            c.setStrokeColorRGB(1, 1, 1)
            c.setLineWidth(.6)
            c.line(x, y, x + 15, y + 7)
            c.line(x, y + 7, x + 15, y)
        elif status == "mixed":
            # Crop the same background, hatch angle, spacing, and stroke as a cell.
            inner_w = (((width - (210 if supplementary else 208)) / 3) - 1) * 75 / 79
            inner_h = ((21 if supplementary else 22.8) - 1) * .9484532
            c.saveState()
            clipping = c.beginPath()
            clipping.rect(x, y, 15, 7)
            c.clipPath(clipping, stroke=0, fill=0)
            c.setStrokeColor(HexColor(COLORS["hatch"]))
            c.setLineWidth(.6)
            for j in range(15):
                end_x = x + j * inner_w * .14666667
                c.line(end_x - inner_w * .98133333, y, end_x, y + inner_h)
            c.restoreState()
        _text(c, x + 22, y - .5, label)


def render_reference_figures(output_dir=ROOT / "figures", data_dir=ROOT / "llm_outputs",
                             result_dir=ROOT / "analysis_results",
                             arrow_pt=9.4, expected_arrow_pt=13.0, knowledge_estimate_pt=STABILITY_PT):
    """Regenerate the reference-sized PDFs using the current analysis results."""
    register_fonts()
    rows = load_rows(data_dir, result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.pdf" for name in ["Fig2", "FigS1"]}
    for name, width, height, kind in [("Fig2", 453.5433, 569.4 + FIG2_EXTRA_PANEL_GAP_PT, "housing"),
                                     ("FigS1", 493.2283, 588, "correlation")]:
        c = canvas.Canvas(str(paths[name]), pagesize=(width, height), pageCompression=1, invariant=1)
        c.setCreator("ReportLab; reference_artwork.py")
        c.setTitle(f"{name}: literature comparison matrix")
        for knowledge in [True, False]:
            # Add whitespace above panel (b) without squeezing rows or the legend.
            offset = 258.2 + FIG2_EXTRA_PANEL_GAP_PT if name == "Fig2" and not knowledge else 0
            _panel(c, width, height, offset, rows[kind], kind, knowledge,
                   arrow_pt, expected_arrow_pt, knowledge_estimate_pt)
            if name == "FigS1" or not knowledge:
                _legend(c, width, name == "FigS1")
                c.showPage()
        c.save()
    return paths
