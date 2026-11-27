"""Export the reference-sized figures without cropping their pages."""
from pathlib import Path
from functools import lru_cache
import os
import shutil
import subprocess

from PIL import Image
from pypdf import PdfReader


@lru_cache(maxsize=None)
def poppler_tool(name):
    """Use an installed Poppler, including the desktop's bundled runtime."""
    installed = shutil.which(name)
    if installed:
        return installed
    cache = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".cache"
    bundled = cache / "codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin"
    candidate = bundled / (name + (".exe" if os.name == "nt" else ""))
    return str(candidate) if candidate.is_file() else None


def export_formats(pdf_path, dpi=600):
    """RGB PNG/TIFF at the requested resolution and vector EPS per page.

    The multipage FigS1.pdf contains both panels. FigS1_page1/2 files export
    the pages separately; no duplicate first-page alias is created.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing figure PDF: {pdf_path}")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError("Raster export resolution must be a positive integer DPI.")
    ppm = poppler_tool("pdftoppm")
    if ppm is None:
        raise RuntimeError("Install Poppler (pdftoppm) to export 600-dpi PNG/TIFF previews.")
    page_count = len(PdfReader(pdf_path).pages)
    result = []
    for page in range(1, page_count + 1):
        stem = pdf_path.stem if page_count == 1 else f"{pdf_path.stem}_page{page}"
        base = pdf_path.with_name(stem)
        render_base = base.with_name(base.name + ".render")
        subprocess.run([ppm, "-f", str(page), "-l", str(page), "-singlefile",
                        "-r", str(dpi), "-png", str(pdf_path), str(render_base)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        png_path = base.with_suffix(".png")
        rendered_path = Path(str(render_base) + ".png")
        with Image.open(rendered_path) as source:
            rgb = source.convert("RGB")
            temporary = base.with_suffix(".tmp.png")
            rgb.save(temporary, dpi=(dpi, dpi))
            temporary.replace(png_path)
            temporary_tiff = base.with_suffix(".tmp.tif")
            rgb.save(temporary_tiff, compression="tiff_lzw", dpi=(dpi, dpi))
            temporary_tiff.replace(base.with_suffix(".tif"))
        rendered_path.unlink()
        result.append(png_path)
        ps = poppler_tool("pdftops")
        if ps:
            subprocess.run([ps, "-f", str(page), "-l", str(page), "-eps",
                            str(pdf_path), str(base.with_suffix(".eps"))], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result
