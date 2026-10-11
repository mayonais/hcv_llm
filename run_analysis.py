"""Run the analysis notebooks from saved inputs, without making LLM API calls."""
import argparse
import contextlib
import hashlib
import gc
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/analysis_runtime/matplotlib_cache"))
local_dependencies = ROOT / "tmp/analysis_runtime/dependencies"
if local_dependencies.exists():
    sys.path.append(str(local_dependencies))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_utils import average_ranks, rank_auc, RESULT_DIR, build_table3

NAMES = ["llm_prompts.ipynb", "llm_outputs_analysis.ipynb",
         "icpsr.ipynb", "tables.ipynb"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_notebook(name):
    notebook = json.loads((ROOT/name).read_text(encoding="utf-8"))
    scope = {"__name__":"__main__"}
    for index,cell in enumerate(notebook["cells"],1):
        if cell["cell_type"]!="code":continue
        source="".join(cell["source"])
        print(f"{name}: cell {index}",flush=True)
        output=io.StringIO()
        try:
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
                exec(compile(source,f"{name}:cell{index}","exec"),scope)
        except Exception:
            cell["outputs"]=[dict(output_type="stream",name="stderr",text=traceback.format_exc().splitlines(keepends=True))]
            (ROOT/name).write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
            raise
        text=output.getvalue()
        cell["execution_count"]=index
        cell["outputs"]=[dict(output_type="stream",name="stdout",text=text.splitlines(keepends=True))] if text else []
        if scope.get("RUN_GENERATION",False):
            raise RuntimeError("This verification run must never enable model generation.")
        plt.close("all")
        # Release renderers between cells; high-resolution plots otherwise remain
        # reachable through notebook variables after pyplot closes their windows.
        for value in scope.values():
            if isinstance(value, matplotlib.figure.Figure):
                if hasattr(value.canvas, "renderer"):
                    value.canvas.renderer = None
        gc.collect()
    (ROOT/name).write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
    return scope

def validate(check_archive=True):
    # Independent tie-rank and pairwise-concordance checks for fallback statistics.
    for sample in [[3,1,1,7,3],[0,0,0,1],[1,4,2,9]]:
        np.testing.assert_array_equal(average_ranks(sample),pd.Series(sample).rank(method="average"))
    for labels,scores in [([0,1,0,1],[1,1,2,3]),([0,0,1,1],[4,3,2,1]),([0,1,0,1],[2,2,2,2])]:
        y=np.asarray(labels);s=np.asarray(scores)
        a=s[y==1][:,None];b=s[y==0][None,:]
        expected=((a>b)+.5*(a==b)).mean()
        np.testing.assert_allclose(rank_auc(y,s),expected,rtol=0,atol=1e-15)
    from scipy.stats import spearmanr as scipy_spearmanr
    from analysis_utils import rank_correlation
    rng=np.random.default_rng(123)
    for _ in range(20):
        x=rng.integers(0,8,70); y=rng.integers(0,12,70)
        np.testing.assert_allclose(rank_correlation(x,y),scipy_spearmanr(x,y).statistic,rtol=1e-12,atol=1e-12)
    summary=pd.read_csv(RESULT_DIR/"literature_summary.csv").set_index("Model")
    expected={"claudesonnet4.6":(22,18,26,31),"deepseek.v3.2":(17,13,25,31),"gpt-oss-120b":(28,19,23,28)}
    for model,values in expected.items():
        row=summary.loc[model]
        assert tuple(row[["N_estimable","N_recovered","N_knowledge_matches","N_prompt_matches"]].astype(int))==values
        assert row.N_directional==28 and row.N_queried==32 and row.N_mixed==4
    housing=pd.read_csv(RESULT_DIR/"housing_effects.csv")
    cols=["Model","Variable","cohens_d","d_ci_lower","d_ci_upper"]
    corr=pd.read_csv(RESULT_DIR/"pooled_outcome_correlations.csv")
    # Local historical comparisons are optional; a fresh clone needs no archive.
    archive=ROOT/"archive"
    if check_archive and archive.exists():
        old=pd.read_csv(archive/"analysis_results/housing_effects.csv")
        pd.testing.assert_frame_equal(housing[cols].reset_index(drop=True),old[cols].reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-12)
        old=pd.read_csv(archive/"analysis_results/pooled_outcome_correlations.csv")
        pd.testing.assert_frame_equal(corr,old,check_exact=False,rtol=1e-12,atol=1e-12)
    abm=pd.read_csv(RESULT_DIR/"abm_mad_pooled.csv")
    assert abm.bootstrap_valid.eq(1000).all() and abm.N_pairs.eq(10).all()
    sweep=pd.read_csv(RESULT_DIR/"abm_mad_tolerance_sweep.csv")
    np.testing.assert_allclose(abm.mean_corr_difference,sweep[sweep.tolerance_years.eq(10)].MAD.to_numpy(),rtol=1e-12)
    for name in ["empirical_mad.csv","empirical_auc.csv"]:
        frame=pd.read_csv(RESULT_DIR/name)
        assert frame.bootstrap_valid.eq(1000).all()
    em=pd.read_csv(RESULT_DIR/"empirical_mad.csv")
    assert em.N_covariates.eq(12).all() and em.bootstrap_covariates_min.eq(12).all() and em.bootstrap_covariates_max.eq(12).all()
    table=build_table3()
    print("\nValidated Table 3\n"+table.to_string(index=False),flush=True)
    if check_archive and archive.exists() and (ROOT/"original_file_hashes.json").exists():
        originals=json.loads((ROOT/"original_file_hashes.json").read_text())
        assert all(sha(archive/name)==digest for name,digest in originals.items())
    # Compare an independently reselected, normalized, non-vaccinated follow-up.
    raw=pd.read_csv(ROOT/"2018-09-13--06.02.24.events.csv",header=4,low_memory=False)
    audit=pd.read_csv(RESULT_DIR/"abm_followup_audit.csv")
    raw=raw[raw.Agent.isin(audit.Agent)].copy()
    raw.HCV=raw.HCV.astype("string").str.strip().str.lower().replace({"infectiousacute":"acute"})
    start=raw[raw.Event.eq("activated")].sort_values(["Agent","Time"]).drop_duplicates("Agent")[["Agent","Time"]].rename(columns={"Time":"start"})
    candidates=raw.merge(start,on="Agent")
    candidates=candidates[candidates.Time.gt(candidates.start)&candidates.HCV.isin(["acute","chronic","susceptible","exposed","recovered"])].copy()
    candidates["error"]=(candidates.Time-candidates.start-.5).abs()
    nearest=candidates.sort_values(["Agent","error","Time"],kind="stable").drop_duplicates("Agent").set_index("Agent")
    np.testing.assert_allclose(audit.set_index("Agent").Time_used.sort_index(),nearest.Time.sort_index())

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Validate saved results without executing notebooks.")
    args=parser.parse_args()
    if args.check:
        validate()
        print("Saved-result checks passed; no model calls were made.",flush=True)
        sys.exit(0)
    start=time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",FutureWarning)
        for name in NAMES:
            run_notebook(name)
            gc.collect()
    validate()
    manifest={"status":"verified","runtime_seconds":time.time()-start,
              "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,
              "bootstrap_resamples":1000,"seed":42,"model_calls":0,
              "source_pdf":"Epidemics_arXiv_source__1_ (2).pdf",
              "inputs":{p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/"outputs/outputs.csv",ROOT/"outputs/knowledge_outputs.csv",ROOT/"outputs/outputs_nyc_hanes.csv",ROOT/"2018-09-13--06.02.24.events.csv"]},
              "code":{name:sha(ROOT/name) for name in NAMES+["analysis_utils.py","association_targets.json","run_analysis.py"]},
              "results":{p.name:sha(p) for p in RESULT_DIR.glob("*.csv")}}
    (RESULT_DIR/"verification_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"All checks passed in {time.time()-start:.1f} seconds; original files unchanged.",flush=True)
