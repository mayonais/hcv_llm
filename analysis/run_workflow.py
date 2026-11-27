"""Execute the paper notebooks from saved responses, with model calls disabled.

Usage: python analysis/run_workflow.py [--figures] [--rebuild-survey-profiles]
Inputs are hashed before and after execution; an execution report records the
completed checks. Use --refresh-notebook-outputs to save captured cell output.
"""
import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

NOTEBOOKS = ["llm_prompts.ipynb", "analysis/llm_outputs_analysis.ipynb",
             "analysis/prompt_sensitivity.ipynb", "analysis/icpsr.ipynb",
             "analysis/invariance.ipynb", "analysis/bootstrap.ipynb"]


def input_hashes():
    paths = [ROOT / "2018-09-13--06.02.24.events.csv"]
    paths.extend(sorted((ROOT / "llm_outputs").glob("*.csv")))
    paths.extend(sorted((ROOT / "llm_outputs").glob("*.jsonl")))
    paths.extend(sorted((ROOT / "ICPSR_31421").glob("**/*.dta")))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class ExecutionFlags(ast.NodeTransformer):
    def __init__(self, rebuild_survey_profiles):
        self.rebuild_survey_profiles = rebuild_survey_profiles

    def visit_Assign(self, node):
        # These are execution switches, never text inside a model prompt.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "RUN_GENERATION":
                node.value = ast.Constant(False)
            elif name == "BUILD_PROFILES":
                node.value = ast.Constant(self.rebuild_survey_profiles)
        return node


def execute_notebook(relative_path, log_dir, rebuild_survey_profiles=False, refresh_outputs=False):
    path = ROOT / relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(path)}
    logs, count = [], 0
    started = time.monotonic()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        text = "".join(cell["source"])
        tree = ast.parse(text, filename=f"{relative_path}:cell{index}")
        tree = ast.fix_missing_locations(ExecutionFlags(rebuild_survey_profiles).visit(tree))
        capture = io.StringIO()
        try:
            with redirect_stdout(capture), redirect_stderr(capture):
                exec(compile(tree, f"{relative_path}:cell{index}", "exec"), namespace)
            if namespace.get("RUN_GENERATION", False):
                raise RuntimeError("Model generation must remain disabled during the workflow.")
        except Exception:
            logs.append(f"CELL {index}\n{capture.getvalue()}\n{traceback.format_exc()}")
            (log_dir / f"{path.stem}.txt").write_text("\n".join(logs), encoding="utf-8")
            raise
        output = capture.getvalue()
        logs.append(f"CELL {index}\n{output}")
        cell["execution_count"] = count
        cell["outputs"] = [dict(output_type="stream", name="stdout", text=output.splitlines(keepends=True))] if output else []
    if refresh_outputs:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (log_dir / f"{path.stem}.txt").write_text("\n".join(logs), encoding="utf-8")
    return dict(notebook=relative_path, cells=count, seconds=round(time.monotonic()-started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="Also regenerate every paper figure and summary table.")
    parser.add_argument("--refresh-notebook-outputs", action="store_true", help="Save captured text outputs into the source notebooks.")
    parser.add_argument("--rebuild-survey-profiles", action="store_true", help="Check the optional NYC HANES construction using the six Stata source files.")
    parser.add_argument("--report", type=Path, default=ROOT / "analysis_results/workflow_check.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    report_path = args.report.resolve()
    log_dir = report_path.parent / "workflow_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    before = input_hashes()
    completed = []
    # Replace any older success before starting; failed runs retain their state.
    report_path.write_text(json.dumps(dict(status="running", notebooks=[]), indent=2)+"\n", encoding="utf-8")
    try:
        for name in NOTEBOOKS + (["figures/figures2.ipynb"] if args.figures else []):
            print(f"Running {name}", flush=True)
            completed.append(execute_notebook(name, log_dir, args.rebuild_survey_profiles, args.refresh_notebook_outputs))
            print(f"Completed {name}: {completed[-1]['seconds']:.1f}s", flush=True)
        from analysis.analysis_utils import build_table3
        build_table3().to_csv(ROOT / "analysis_results/table3.csv", index=False)
        after = input_hashes()
        if before != after:
            raise RuntimeError("Raw study inputs changed during the workflow.")
    except Exception as error:
        report_path.write_text(json.dumps(dict(status="failed", notebooks=completed,
            error=f"{type(error).__name__}: {error}", raw_inputs_unchanged=before==input_hashes()), indent=2)+"\n", encoding="utf-8")
        raise
    report = dict(status="completed", notebooks=completed, raw_inputs_unchanged=True,
                  input_hashes=before, generation_enabled=False,
                  survey_profiles_rebuilt=args.rebuild_survey_profiles,
                  figures_regenerated=args.figures)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"All notebooks completed; raw study inputs unchanged. Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
