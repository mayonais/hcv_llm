"""Shared definitions and reporting for the paper analyses.

Definitions follow the study's methods and association benchmarks.
The raw CSVs are never modified. Direction scoring uses unrounded estimates.
"""
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "llm_outputs"
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
PERCENTAGE_POINT_OUTCOMES = {"Public_prop", "Recept_sharing", "Partner_HCV_prop", "Treatment", "HCV_active"}
HOUSING_UNITS = {
    "Injections": "injections/week", "Drug_in": "people/week", "Drug_out": "people/week",
    "Partners": "partners/week", "Alt_route": "occasions/week", "Overdose": "overdoses/week",
    **{variable: "pp" for variable in PERCENTAGE_POINT_OUTCOMES},
}
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
    if set(frame.Model) != set(MODELS):
        raise ValueError("Knowledge records must include exactly the three evaluated models.")
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

def paired_housing_effect(frame, outcome_col):
    """Mean homeless-minus-stable difference, without confidence intervals.

    `estimate` uses the reporting unit (percentage points for proportions and
    binary outcomes); `mean_difference` retains the original outcome scale.
    Zero and constant paired differences remain estimable. Directional
    stability and estimability are calculated separately in bootstrap.ipynb.
    """
    keys = ["Agent", "Homeless"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Each agent/housing identifier must be unique and nonmissing.")
    if not frame.Homeless.isin([0, 1]).all():
        raise ValueError("Housing conditions must be 0 (stable) or 1 (homeless).")
    wide = frame.pivot(index="Agent", columns="Homeless", values=outcome_col).reindex(columns=[0, 1])
    total = len(wide)
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna()
    delta = (wide[1] - wide[0]).to_numpy(dtype=float)
    n = len(delta)
    point = float(delta.mean()) if n else np.nan
    constant = bool(n and np.ptp(delta) == 0)
    scale = 100 if outcome_col in PERCENTAGE_POINT_OUTCOMES else 1
    return dict(N_pairs=n, N_excluded=total-n,
                mean_stable=float(wide[0].mean()) if n else np.nan,
                mean_homeless=float(wide[1].mean()) if n else np.nan,
                mean_difference=point, estimate=scale*point,
                effect_measure="percentage_point_difference" if scale == 100 else "mean_paired_difference",
                unit=HOUSING_UNITS[outcome_col], scale=scale,
                estimable=bool(np.isfinite(point)), constant_differences=constant)


def score_literature(housing, correlations, knowledge):
    """Score original-prompt outputs; wording comparisons live in their notebook."""
    knowledge = normalize_knowledge(knowledge)
    records, summaries = [], []
    for model in MODELS:
        for t in TARGETS:
            pair = (t["exposure"],t["outcome"])
            a,b = COLUMNS[pair[0]],COLUMNS[pair[1]]
            if a == "Homeless":
                rows = housing.loc[housing.Model.eq(model)&housing.Variable.eq(b)&housing.Paraphrase.eq(0)]
                value_column = "estimate"
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
            k = knowledge.loc[knowledge.Model.eq(model)&knowledge.exposure.eq(pair[0])&knowledge.outcome.eq(pair[1])&knowledge.Paraphrase.eq(0)]
            direction = k.iloc[0].direction
            scored = t["expected"] != "↔"
            sign = {"↑":1,"↓":-1,"↔":0}[t["expected"]]
            records.append(dict(Model=model,kind=kind,exposure=pair[0],outcome=pair[1],
                expected=t["expected"],scored=scored,estimate=value,estimable=bool(np.isfinite(value)),
                generated_match=bool(scored and np.isfinite(value) and np.sign(value)==sign),
                knowledge_direction=direction,
                knowledge_match=bool(scored and direction=={1:"increase",-1:"decrease",0:"mixed"}[sign])))
        audit = pd.DataFrame([r for r in records if r["Model"]==model])
        scored = audit.loc[audit.scored]
        n,est,matched,know = len(scored),int(scored.estimable.sum()),int(scored.generated_match.sum()),int(scored.knowledge_match.sum())
        assert n==28
        summaries.append(dict(Model=model,N_queried=32,N_mixed=4,N_directional=n,
            N_estimable=est,N_recovered=matched,estimable_pct=100*est/n,
            conditional_pct=100*matched/est if est else np.nan,unconditional_pct=100*matched/n,
            N_knowledge_matches=know,knowledge_pct=100*know/n,
            knowledge_generation_gap_pp=100*(know-matched)/n))
    return pd.DataFrame(records),pd.DataFrame(summaries)

STRATA_COLUMNS = ["Gender", "Race", "Experience_group"]
EXPECTED_PROFILES = 224

def reconstruct_original_profiles(cnep_path, expected_agents):
    columns = ["Agent", "Time", "HCV", "Age", "Age_Started", "Gender", "Race", "Daily_injection_intensity"]
    source = pd.read_csv(cnep_path, header=4, usecols=columns)
    source["HCV"] = source.HCV.astype("string").str.strip().str.lower().replace({"infectiousacute": "acute"})
    source = source.drop(index=[31330, 4816], errors="ignore")
    clean = source.loc[~source.HCV.isin(["vaccinated", "exposed"])].copy()
    baseline = clean.sort_values(["Time"]).drop_duplicates("Agent", keep="first")
    baseline = baseline.loc[baseline.HCV.eq("susceptible")].copy()
    baseline["Injection_Experience"] = baseline.Age - baseline.Age_Started
    event_counts = clean.groupby("Agent").Time.nunique()
    valid_agents = event_counts.loc[event_counts.ge(2)].index
    baseline = baseline.loc[baseline.Age.ge(18) & baseline.Daily_injection_intensity.gt(0)
                            & baseline.Agent.isin(valid_agents)].copy()
    young = baseline.loc[baseline.Injection_Experience.lt(3)].copy()
    young["Experience_group"] = "Young"
    old = baseline.loc[baseline.Injection_Experience.ge(3)].copy()
    old["Experience_group"] = "Old"
    eligible = pd.concat([young, old])
    first = pd.concat([group.sample(7, random_state=42)
                       for _, group in eligible.groupby(STRATA_COLUMNS, sort=True)])
    remaining = eligible.loc[~eligible.Agent.isin(set(first.Agent))]
    second = pd.concat([group.sample(n=min(7, len(group)), random_state=99)
                        for _, group in remaining.groupby(STRATA_COLUMNS, sort=True)])
    selected = pd.concat([first, second])
    if selected.Agent.nunique() != EXPECTED_PROFILES or selected.Agent.duplicated().any():
        raise ValueError("Original sampling no longer produces 224 distinct profiles.")
    if set(selected.Agent) != set(expected_agents):
        raise ValueError("Reconstructed profile IDs differ from the saved-output cohort.")
    counts = selected.groupby(STRATA_COLUMNS, sort=True).size()
    if len(counts) != 16 or not counts.eq(14).all():
        raise ValueError("Expected 16 original strata of 14 profiles each.")
    return selected[["Agent"] + STRATA_COLUMNS + ["Injection_Experience"]].sort_values(
        STRATA_COLUMNS + ["Agent"], kind="stable").reset_index(drop=True)


def stratified_profile_draws(profile_frame, n_resamples, seed):
    if n_resamples < 1:
        raise ValueError("At least one bootstrap resample is required.")
    profile_frame = profile_frame.reset_index(drop=True)
    if profile_frame.empty or profile_frame[STRATA_COLUMNS].isna().any().any():
        raise ValueError("Nonempty profiles and complete original stratum labels are required.")
    groups = [group.index.to_numpy() for _, group in profile_frame.groupby(STRATA_COLUMNS, sort=True)]
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(indices, size=(n_resamples, len(indices)), replace=True)
                            for indices in groups], axis=1)
    # Every replicate preserves each original stratum count, counting duplicates.
    codes = np.empty(len(profile_frame), dtype=int)
    for code, indices in enumerate(groups):
        codes[indices] = code
        assert np.all((codes[indices] == code))
    selected_codes = codes[draws]
    for code, indices in enumerate(groups):
        assert np.all(np.sum(selected_codes == code, axis=1) == len(indices))
    return draws


def paired_cnep_mad(frame,cols,B=1000,seed=42,*,profiles):
    """Percentile MAD CI from original-stratum draws of paired profiles."""
    frame = frame.sort_values(["Agent","Homeless"]).copy()
    if frame.duplicated(["Agent","Homeless"]).any():
        raise ValueError("Duplicate agent/housing rows.")
    conditions=frame.groupby("Agent").Homeless.agg(set)
    if not conditions.map(lambda s:s=={0,1}).all():
        raise ValueError("Each agent needs both housing conditions.")
    n=len(conditions)
    values=frame[cols+["CNEP_"+v for v in cols]].to_numpy(float).reshape(n,2,2*len(cols))
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
        raise ValueError("No eligible CNEP+ correlation pairs.")
    point=np.abs(corr[eligible,0]-corr[eligible,1]).mean()
    if profiles.Agent.duplicated().any() or set(profiles.Agent) != set(conditions.index):
        raise ValueError("Stratum profiles must match the paired MAD cohort exactly.")
    aligned_profiles = profiles.set_index("Agent").loc[conditions.index].reset_index()
    counts = aligned_profiles.groupby(STRATA_COLUMNS, sort=True).size()
    if len(counts) != 16 or not counts.eq(14).all():
        raise ValueError("CNEP+ requires the original 16 strata of 14 profiles.")
    selected_profiles = stratified_profile_draws(aligned_profiles, B, seed)
    draws=[]
    for selected in selected_profiles:
        boot=pair_vectors(values[selected])[eligible]
        draws.append(np.abs(boot[:,0]-boot[:,1]).mean() if np.isfinite(boot).all() else np.nan)
    draws=np.asarray(draws);valid=np.isfinite(draws)
    lo,hi=np.percentile(draws,[2.5,97.5]) if valid.all() else (np.nan,np.nan)
    return dict(Housing="Combined stable and homeless",N_agents=n,N_rows=2*n,
        N_pairs=int(eligible.sum()),N_pairs_prespecified=len(pairs),mean_corr_difference=point,
        ci_lower=lo,ci_upper=hi,bootstrap_valid=int(valid.sum()),bootstrap_total=B,
        ci_status="ok" if valid.all() else "undefined replicates; CI withheld",
        Bootstrap_strata="Gender x Race x injection experience",
        N_strata=len(counts), N_profiles_per_stratum=int(counts.iloc[0]),
        Bootstrap_seed=seed)

def _bootstrap_stability_line(bootstrap, model, pair, value, kind):
    """Validate matching original-prompt results and format conditional agreement."""
    rows = bootstrap.loc[bootstrap.Kind.eq(kind) & bootstrap.Paraphrase.eq(0)
        & bootstrap.Model.eq(model) & bootstrap.Exposure.eq(pair[0]) & bootstrap.Outcome.eq(pair[1])]
    if len(rows) != 1:
        raise ValueError(f"Run analysis/bootstrap.ipynb: expected one {kind} summary for {model}/{pair}.")
    boot = rows.iloc[0]
    if (boot.Expected_direction != EXPECTED[pair] or boot.B != 1000
            or bool(boot.Observed_estimable) != bool(np.isfinite(value))
            or not np.isclose(value, boot.Observed_estimate, rtol=0, atol=1e-12, equal_nan=True)):
        raise ValueError(f"Stale {kind} bootstrap for {model}/{pair}; rerun analysis/bootstrap.ipynb.")
    if (boot.N_estimable + boot.N_nonestimable != boot.B
            or not 0 <= boot.N_estimable <= boot.B
            or not np.isclose(boot.Estimable_pct, 100*boot.N_estimable/boot.B, rtol=0, atol=1e-12)):
        raise ValueError(f"Invalid estimability denominator for {model}/{pair}.")
    if EXPECTED[pair] == "↔" or boot.N_estimable == 0:
        if pd.notna(boot.Directional_stability_pct):
            raise ValueError(f"Bootstrap directional agreement must be missing for {model}/{pair}.")
        return "agreement n/a"
    if (boot.N_direction_assessed != boot.N_estimable
            or not 0 <= boot.N_agree <= boot.N_estimable
            or boot.N_agree + boot.N_disagree != boot.N_estimable
            or not np.isclose(boot.Directional_stability_pct, 100*boot.N_agree/boot.N_estimable, rtol=0, atol=1e-12)):
        raise ValueError(f"Invalid bootstrap directional-agreement denominator for {model}/{pair}.")
    return f"agreement {boot.Directional_stability_pct:.1f}%"


def housing_figure_values(frame, bootstrap):
    """Observed effects plus original-prompt bootstrap directional agreement.

    Agreement is literature agreement among estimable draws. Estimability
    remains in the bootstrap results. Mixed expectations have no score.
    """
    out={}
    for pair in expected_directions("housing"):
        v=COLUMNS[pair[1]];strings=[]
        for model in MODELS:
            rows=frame.loc[frame.Model.eq(model)&frame.Variable.eq(v)&frame.Paraphrase.eq(0)]
            if len(rows) != 1:
                raise ValueError(f"Expected exactly one housing result for {model}/{v}.")
            row=rows.iloc[0]
            value=row.estimate
            arrow="↑" if value>0 else "↓" if value<0 else "—"
            # Show the small negative DeepSeek injection difference in Fig2.
            decimals = 3 if (model, v) == ("deepseek.v3.2", "Injections") else 2
            text=f"{arrow} {value:.{decimals}f}" if np.isfinite(value) else "NaN"
            text+="\n"+_bootstrap_stability_line(bootstrap, model, pair, value, "housing")
            strings.append(text)
        out[pair]=strings
    return out

def correlation_figure_values(frame, bootstrap):
    """Pooled correlations plus original-prompt bootstrap directional agreement."""
    out={}
    for pair in expected_directions("correlation"):
        a,b=COLUMNS[pair[0]],COLUMNS[pair[1]];values=[]
        for model in MODELS:
            rows=frame.loc[frame.model.eq(model)&((frame.var1.eq(a)&frame.var2.eq(b))|(frame.var1.eq(b)&frame.var2.eq(a)))]
            if len(rows)!=1:raise ValueError(f"Missing correlation for {model}/{pair}")
            value=float(rows.iloc[0].correlation)
            arrow="↑" if value>0 else "↓" if value<0 else "—"
            # Preserve the small nonzero GPT-OSS treatment/HCV correlation in FigS1.
            small_hcv = (model == "gpt-oss-120b" and {a, b} == {"Treatment", "HCV_active"}
                         and value != 0 and f"{value:.2f}" in {"0.00", "-0.00"})
            decimals = 3 if small_hcv else 2
            text=f"{arrow} {value:.{decimals}f}" if np.isfinite(value) else "NaN"
            values.append(text+"\n"+_bootstrap_stability_line(bootstrap, model, pair, value, "correlation"))
        out[pair]=values
    return out

def build_table3():
    """Combine main comparisons with the independently run prompt sensitivity."""
    for name in ["prompt_semantic.csv", "knowledge_prompt_agreement.csv"]:
        if not (RESULT_DIR/name).is_file():
            raise FileNotFoundError(f"Run analysis/prompt_sensitivity.ipynb to generate {name} before Table 3.")
    summary=pd.read_csv(RESULT_DIR/"literature_summary.csv").set_index("Model")
    cnep=pd.read_csv(RESULT_DIR/"cnep_mad_pooled.csv").set_index("Model")
    nyc_hanes=pd.read_csv(RESULT_DIR/"nyc_hanes_mad.csv").set_index("Model")
    auc=pd.read_csv(RESULT_DIR/"nyc_hanes_auc.csv").set_index("Model")
    sensitivity=pd.read_csv(RESULT_DIR/"prompt_semantic.csv")
    knowledge_prompt_agreement=pd.read_csv(RESULT_DIR/"knowledge_prompt_agreement.csv").set_index("Model")
    rows=[]
    def proportion(n,N):
        return f"{int(n):,}/{int(N):,} ({100*n/N:.1f}%)" if N else f"{int(n):,}/0 (n/a)"
    def interval(x,lo,hi):return f"{x:.2f} ({lo:.2f}–{hi:.2f})"
    metrics=["Estimable targets","Conditional directional recovery","Unconditional directional recovery",
        "Literature-informed directional agreement","CNEP+ correlation MAD","NYC HANES correlation MAD","NYC HANES HCV risk-score AUC",
        "Housing-difference sign agreement","Agreement between prompt versions"]
    for metric in metrics:
        row={"Metric":metric}
        for model in MODELS:
            r=summary.loc[model]
            if metric==metrics[0]:text=proportion(r.N_estimable,r.N_directional)
            elif metric==metrics[1]:text=proportion(r.N_recovered,r.N_estimable)
            elif metric==metrics[2]:text=proportion(r.N_recovered,r.N_directional)
            elif metric==metrics[3]:text=proportion(r.N_knowledge_matches,r.N_directional)
            elif metric==metrics[4]:
                b=cnep.loc[model];text=interval(b.mean_corr_difference,b.ci_lower,b.ci_upper)
            elif metric==metrics[5]:
                b=nyc_hanes.loc[model];text=interval(b.NYC_HANES_MAD,b.MAD_lower_95,b.MAD_upper_95)
            elif metric==metrics[6]:
                b=auc.loc[model];text=interval(b.AUC,b.AUC_lower_95,b.AUC_upper_95)
            elif metric==metrics[7]:
                b=sensitivity.loc[sensitivity.model.eq(model)]
                text=proportion(round((b.N*b.directional_consistency).sum()),b.N.sum())
            else:
                b=knowledge_prompt_agreement.loc[model];text=proportion(b.agreement,b.total)
            row[MODEL_LABELS[model]]=text
        rows.append(row)
    return pd.DataFrame(rows)
