"""Shared paper definitions and reporting for the copied notebooks.

Source: Epidemics_arXiv_source__1_ (2).pdf, Tables 2, 3, and S1.
The raw CSVs are never modified. Direction scoring uses unrounded estimates.
"""
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs"
RESULT_DIR = ROOT / "analysis_results"
FIGURE_DIR = ROOT / "figures"
RESULT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
MODELS = ["claudesonnet4.6", "deepseek.v3.2", "gpt-oss-120b"]
MODEL_LABELS = dict(zip(MODELS, ["Claude Sonnet 4.6", "DeepSeek v3.2", "GPT-OSS-120B"]))
TARGETS = json.loads((ROOT / "association_targets.json").read_text(encoding="utf-8"))
PAPER_PAIRS = [(t["exposure"], t["outcome"]) for t in TARGETS]
EXPECTED = {pair: t["expected"] for pair, t in zip(PAPER_PAIRS, TARGETS)}
TERMS = {
    "Homeless": "current or recent homelessness", "Public_prop": "public injecting",
    "Overdose": "non-fatal overdose incidence", "Recept_sharing": "receptive syringe sharing",
    "Injections": "injection frequency", "Treatment": "engaging in opioid use disorder treatment",
    "HCV_active": "personal HCV infection status (RNA-positive)",
    "Alt_route": "non-injection routes of opioid administration (smoking/snorting)",
    "Partners": "number of injection partners", "Partner_HCV_prop": "number of HCV-positive injection partners",
}
COLUMNS = {v: k for k, v in TERMS.items()}
assert len(PAPER_PAIRS) == 32 and len(set(PAPER_PAIRS)) == 32
assert sum(t["expected"] == "↔" for t in TARGETS) == 4
assert not any("fentanyl" in " ".join(p).lower() for p in PAPER_PAIRS)

def average_ranks(x):
    """One-based average ranks, including ties."""
    _, inverse, counts = np.unique(np.asarray(x), return_inverse=True, return_counts=True)
    ends = counts.cumsum()
    return ((ends - counts + 1 + ends) / 2.0)[inverse]

def rank_correlation(x, y):
    values = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 3 or any(np.unique(values[:,j]).size < 2 for j in [0,1]):
        return np.nan
    ranks = np.column_stack([average_ranks(values[:,0]), average_ranks(values[:,1])])
    ranks -= ranks.mean(axis=0)
    return float(np.sum(ranks[:,0]*ranks[:,1]) / np.sqrt(np.sum(ranks[:,0]**2)*np.sum(ranks[:,1]**2)))

def rank_auc(y, scores):
    """Mann--Whitney AUC; tied scores receive half credit."""
    y, scores = np.asarray(y, float), np.asarray(scores, float)
    if not np.isfinite(scores).all() or not np.isin(y,[0,1]).all():
        raise ValueError("AUC requires finite scores and binary labels.")
    n1, n0 = int((y==1).sum()), int((y==0).sum())
    if not n1 or not n0:
        raise ValueError("AUC requires both antibody-status groups.")
    ranks = average_ranks(scores)
    return float((ranks[y==1].sum() - n1*(n1+1)/2) / (n1*n0))

# Use the original libraries when installed. The fallbacks implement only the
# rank statistics needed by these notebooks; they do not provide p-values.
try:
    from scipy.stats import spearmanr
except ImportError:
    def spearmanr(x,y):
        return SimpleNamespace(statistic=rank_correlation(x,y), pvalue=np.nan)
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = rank_auc

def normalize_knowledge(frame):
    frame = frame.copy()
    for target, alternate in [("exposure","factor1"),("outcome","factor2")]:
        if target not in frame:
            frame[target] = np.nan
        if alternate in frame:
            frame[target] = frame[target].fillna(frame[alternate])
        frame[target] = frame[target].astype("string").str.strip()
    keys = list(zip(frame.exposure,frame.outcome))
    frame = frame.loc[[p in EXPECTED for p in keys]].copy()
    frame["direction"] = frame.direction.astype("string").str.strip().str.lower()
    keycols = ["Model","exposure","outcome","Paraphrase"]
    if frame[keycols].isna().any().any() or frame.duplicated(keycols).any():
        raise ValueError("Knowledge records must have unique nonmissing identifiers.")
    if not frame.direction.isin(["increase","decrease","mixed","uncertain"]).all():
        raise ValueError("Unknown knowledge direction.")
    if not frame.Paraphrase.isin([0,1]).all():
        raise ValueError("Knowledge prompt versions must be 0 and 1.")
    for (model, version), group in frame.groupby(["Model","Paraphrase"]):
        if set(zip(group.exposure,group.outcome)) != set(PAPER_PAIRS):
            raise ValueError(f"{model}/{version}: expected exactly the paper's 32 pairs.")
    for model, group in frame.groupby("Model"):
        if set(group.Paraphrase) != {0,1}:
            raise ValueError(f"{model}: both knowledge wordings are required.")
    return frame

def expected_directions(kind):
    return {pair: direction for pair,direction in EXPECTED.items()
            if (pair[0]==TERMS["Homeless"]) == (kind=="housing")}

def score_literature(housing, correlations, knowledge):
    knowledge = normalize_knowledge(knowledge)
    records, summaries = [], []
    for model in MODELS:
        for t in TARGETS:
            pair = (t["exposure"],t["outcome"])
            a,b = COLUMNS[pair[0]],COLUMNS[pair[1]]
            if a == "Homeless":
                rows = housing.loc[housing.Model.eq(model)&housing.Variable.eq(b)&housing.Paraphrase.eq(0)]
                value_column = "cohens_d"
                kind = "housing"
            else:
                rows = correlations.loc[correlations.model.eq(model)&
                    ((correlations.var1.eq(a)&correlations.var2.eq(b))|
                     (correlations.var1.eq(b)&correlations.var2.eq(a)))]
                value_column = "correlation"
                kind = "correlation"
            if len(rows)!=1:
                raise ValueError(f"Expected one result for {model}/{pair}.")
            value = float(rows.iloc[0][value_column])
            k = knowledge.loc[knowledge.Model.eq(model)&knowledge.exposure.eq(pair[0])&knowledge.outcome.eq(pair[1])]
            directions = k.set_index("Paraphrase").direction
            scored = t["expected"] != "↔"
            sign = {"↑":1,"↓":-1,"↔":0}[t["expected"]]
            records.append(dict(Model=model,kind=kind,exposure=pair[0],outcome=pair[1],
                expected=t["expected"],scored=scored,estimate=value,estimable=bool(np.isfinite(value)),
                generated_match=bool(scored and np.isfinite(value) and np.sign(value)==sign),
                knowledge_direction=directions[0],
                knowledge_match=bool(scored and directions[0]=={1:"increase",-1:"decrease",0:"mixed"}[sign]),
                paraphrase_direction=directions[1],prompt_match=bool(directions[0]==directions[1])))
        audit = pd.DataFrame([r for r in records if r["Model"]==model])
        scored = audit.loc[audit.scored]
        n,est,matched,know = len(scored),int(scored.estimable.sum()),int(scored.generated_match.sum()),int(scored.knowledge_match.sum())
        assert n==28
        summaries.append(dict(Model=model,N_queried=32,N_mixed=4,N_directional=n,
            N_estimable=est,N_recovered=matched,estimable_pct=100*est/n,
            conditional_pct=100*matched/est,unconditional_pct=100*matched/n,
            N_knowledge_matches=know,knowledge_pct=100*know/n,
            knowledge_generation_gap_pp=100*(know-matched)/n,
            N_prompt_matches=int(audit.prompt_match.sum()),N_prompt_pairs=32,
            knowledge_prompt_pct=100*audit.prompt_match.mean()))
    return pd.DataFrame(records),pd.DataFrame(summaries)

def paired_abm_mad(frame,cols,B=1000,seed=42):
    frame = frame.sort_values(["Agent","Homeless"]).copy()
    if frame.duplicated(["Agent","Homeless"]).any():
        raise ValueError("Duplicate agent/housing rows.")
    conditions=frame.groupby("Agent").Homeless.agg(set)
    if not conditions.map(lambda s:s=={0,1}).all():
        raise ValueError("Each agent needs both housing conditions.")
    n=len(conditions)
    values=frame[cols+["ABM_"+v for v in cols]].to_numpy(float).reshape(n,2,2*len(cols))
    pairs=list(combinations(range(len(cols)),2));p=len(cols)
    def pair_vectors(data):
        out=[]
        data=data.reshape(-1,2*p)
        for j,k in pairs:
            subset=data[:,[j,k,p+j,p+k]]
            subset=subset[np.isfinite(subset).all(axis=1)]
            out.append([rank_correlation(subset[:,0],subset[:,1]),rank_correlation(subset[:,2],subset[:,3])])
        return np.asarray(out)
    corr=pair_vectors(values);eligible=np.isfinite(corr).all(axis=1)
    if not eligible.any():
        raise ValueError("No eligible ABM correlation pairs.")
    point=np.abs(corr[eligible,0]-corr[eligible,1]).mean()
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(B):
        # Equivalent to sampling sorted agent IDs and retaining both rows.
        selected=rng.choice(n,size=n,replace=True)
        boot=pair_vectors(values[selected])[eligible]
        draws.append(np.abs(boot[:,0]-boot[:,1]).mean() if np.isfinite(boot).all() else np.nan)
    draws=np.asarray(draws);valid=np.isfinite(draws)
    lo,hi=np.percentile(draws,[2.5,97.5]) if valid.all() else (np.nan,np.nan)
    return dict(Housing="Combined stable and homeless",N_agents=n,N_rows=2*n,
        N_pairs=int(eligible.sum()),N_pairs_prespecified=len(pairs),mean_corr_difference=point,
        ci_lower=lo,ci_upper=hi,bootstrap_valid=int(valid.sum()),bootstrap_total=B,
        ci_status="ok" if valid.all() else "undefined replicates; CI withheld")

def housing_figure_values(frame):
    out={}
    for pair in expected_directions("housing"):
        v=COLUMNS[pair[1]];strings=[]
        for model in MODELS:
            row=frame.loc[frame.Model.eq(model)&frame.Variable.eq(v)&frame.Paraphrase.eq(0)].iloc[0]
            value=row.cohens_d
            if not np.isfinite(value):
                strings.append("NaN");continue
            arrow="↑" if value>0 else "↓" if value<0 else "—"
            text=f"{arrow} {value:.2f}"
            if np.isfinite(row.d_ci_lower) and np.isfinite(row.d_ci_upper):
                text+=f" ({row.d_ci_lower:.2f}–{row.d_ci_upper:.2f})"
            strings.append(text)
        out[pair]=strings
    return out

def correlation_figure_values(frame):
    out={}
    for pair in expected_directions("correlation"):
        a,b=COLUMNS[pair[0]],COLUMNS[pair[1]];values=[]
        for model in MODELS:
            rows=frame.loc[frame.model.eq(model)&((frame.var1.eq(a)&frame.var2.eq(b))|(frame.var1.eq(b)&frame.var2.eq(a)))]
            if len(rows)!=1:raise ValueError(f"Missing correlation for {model}/{pair}")
            value=float(rows.iloc[0].correlation);values.append(value if np.isfinite(value) else None)
        out[pair]=values
    return out

def build_table3():
    summary=pd.read_csv(RESULT_DIR/"literature_summary.csv").set_index("Model")
    abm=pd.read_csv(RESULT_DIR/"abm_mad_pooled.csv").set_index("Model")
    empirical=pd.read_csv(RESULT_DIR/"empirical_mad.csv").set_index("Model")
    auc=pd.read_csv(RESULT_DIR/"empirical_auc.csv").set_index("Model")
    sensitivity=pd.read_csv(RESULT_DIR/"prompt_semantic.csv")
    rows=[]
    def proportion(n,N):return f"{int(n):,}/{int(N):,} ({100*n/N:.1f}%)"
    def interval(x,lo,hi):return f"{x:.2f} ({lo:.2f}–{hi:.2f})"
    metrics=["Estimable targets","Conditional directional recovery","Unconditional directional recovery",
        "Knowledge-probe agreement","ABM correlation MAD","Survey correlation MAD","HCV-score AUC",
        "Housing-direction agreement","Knowledge-probe stability"]
    for metric in metrics:
        row={"Metric":metric}
        for model in MODELS:
            r=summary.loc[model]
            if metric==metrics[0]:text=proportion(r.N_estimable,r.N_directional)
            elif metric==metrics[1]:text=proportion(r.N_recovered,r.N_estimable)
            elif metric==metrics[2]:text=proportion(r.N_recovered,r.N_directional)
            elif metric==metrics[3]:text=proportion(r.N_knowledge_matches,r.N_directional)
            elif metric==metrics[4]:
                b=abm.loc[model];text=interval(b.mean_corr_difference,b.ci_lower,b.ci_upper)
            elif metric==metrics[5]:
                b=empirical.loc[model];text=interval(b.Empirical_MAD,b.MAD_lower_95,b.MAD_upper_95)
            elif metric==metrics[6]:
                b=auc.loc[model];text=interval(b.AUC,b.AUC_lower_95,b.AUC_upper_95)
            elif metric==metrics[7]:
                b=sensitivity.loc[sensitivity.model.eq(model)]
                text=proportion(round((b.N*b.directional_consistency).sum()),b.N.sum())
            else:text=proportion(r.N_prompt_matches,r.N_prompt_pairs)
            row[MODEL_LABELS[model]]=text
        rows.append(row)
    return pd.DataFrame(rows)
