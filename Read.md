# HCV LLM evaluation

This repository implements the evaluation described in the main manuscript and supplementary information: literature-informed directional recovery, bootstrap directional agreement and estimability, sensitivity to prompt wording, and supplementary comparisons with CNEP+ simulated follow-up data and NYC HANES HCV risk scores. The models are Claude Sonnet 4.6, DeepSeek v3.2, and GPT-OSS-120B.

## Active workflow

| File | Purpose |
| --- | --- |
| `llm_prompts.ipynb` | Reconstruct the original 224-profile CNEP+ cohort and construct housing-scenario and knowledge-probe prompts. Model generation is optional and disabled by default. |
| `analysis/llm_outputs_analysis.ipynb` | Original-prompt mean paired housing differences, pooled generated-outcome correlations, directional recovery, matched CNEP+ correlation MAD, and follow-up timing sensitivity. |
| `analysis/prompt_sensitivity.ipynb` | Housing-difference sign agreement between original and paraphrased behavioral prompts, and exact direction-category agreement between knowledge-probe prompt versions. |
| `analysis/icpsr.ipynb` | NYC HANES correlation MAD and HCV risk-score AUC from saved responses. Optional reconstruction from the six ICPSR Stata files is disabled by default. |
| `analysis/invariance.ipynb` | Within-condition invariance for the eleven evaluated outcomes, paired differences, and zero-mean cancellation diagnostics. |
| `analysis/bootstrap.ipynb` | Stratified paired-profile bootstrap directional agreement and estimability for the 32 relationships. |
| `analysis/analysis_utils.py` | Shared target definitions, rank statistics, paired CNEP+ MAD bootstrap, scoring, and Table 3 formatting. |
| `analysis/artwork.py` | FigS2 and FigS3 drawing and export. |
| `analysis/run_workflow.py` | Execute the active notebooks with model requests disabled, record logs, and verify that raw inputs remain unchanged. |
| `figures/figures2.ipynb` | Regenerate Fig2, FigS1, FigS2, FigS3, Table 3, Table S2, and Table S3. |
| `figures/reference_artwork.py` | Editable vector drawing for Fig2 and FigS1. |
| `figures/figure_exports.py` | Full-page PDF-to-PNG/TIFF/EPS exports using Poppler. |
| `association_targets.json` | The 32 prespecified relationships and their literature-informed directions. |

## Setup and execution

Install the required packages with a working Python installation:

```sh
python -m pip install -r requirements.txt
```

Run the analysis from the repository root:

```sh
python analysis/run_workflow.py
```

To also regenerate every paper figure and summary table:

```sh
python analysis/run_workflow.py --figures
```

To check optional NYC HANES profile reconstruction from the original Stata files:

```sh
python analysis/run_workflow.py --rebuild-survey-profiles
```

The runner disables model generation, executes every code cell in each selected notebook, captures text logs, refreshes Table 3, and hashes the saved responses and source data before and after. Its default report is `analysis_results/workflow_check.json`; logs are in `analysis_results/workflow_logs/`. `--refresh-notebook-outputs` also saves captured text outputs in the notebooks. Failed runs replace any previous success report with a failure report.

The notebooks can also be run interactively in this order: prompts, outcome analysis, prompt sensitivity, NYC HANES, invariance, bootstrap, figures. Analysis notebooks accept either the repository root or `analysis/` as their working directory; the figure notebook also accepts `figures/`. Figure generation requires Poppler (`pdftoppm`); EPS alternatives additionally require `pdftops`. The figure notebook enables both `BUILD_OTHER_FIGURES` and `BUILD_SUMMARY_TABLES` by default. Either switch can be set to `False` to skip those optional exports during an interactive run.

## Study inputs and preserved schemas

- `llm_outputs/outputs.csv` and `.jsonl`: saved behavioral responses, identified by model, agent, housing condition, and prompt version.
- `llm_outputs/knowledge_outputs.csv` and `.jsonl`: saved knowledge-probe responses; analysis selects the 32 retained relationships.
- `llm_outputs/outputs_nyc_hanes.csv` and `.jsonl`: participant characteristics, observed antibody status, and generated risk scores.
- `2018-09-13--06.02.24.events.csv`: CNEP+ simulation records, read with `header=4`.

Saved response keys, API model identifiers, historical prompt-version labels, and literal prompt strings are preserved. Behavioral `Paraphrase=0` is original and `2` is the broad paraphrase; the earlier behavioral `1` remains in the raw files but is excluded from the paper analyses. Knowledge probes use their own original/alternate labels `0` and `1`.

The optional NYC HANES construction uses `ICPSR_31421/DS0001/31421-0001-Data.dta` through `DS0006/31421-0006-Data.dta`. These source downloads are not tracked. The saved-score analysis runs without them. In `analysis/icpsr.ipynb`, set `BUILD_PROFILES=True` to reconstruct profiles; generation also requires `RUN_GENERATION=True`. Generation uses Boto3 and Amazon Bedrock credentials and writes raw outputs only when explicitly enabled in the generation notebooks. `boto3` is not required to analyze saved responses.

## Definitions matched to the paper

- The 224 original profiles occupy 16 sex, race, and injection-experience strata, with 14 profiles per stratum. Injection experience is less than 3 versus at least 3 years. The internal `Experience_group` values `Young` and `Old` describe injection experience, not chronological age. Original strata are reconstructed from unrounded CNEP+ data.
- Housing comparisons are mean within-profile differences, homelessness minus stable housing. Count outcomes use average weekly units for the final four weeks of the six-month scenario. Proportions and binary outcomes are reported as percentage-point differences. Zero paired differences remain estimable; a zero estimate does not match an expected increase or decrease.
- Generated-outcome Spearman correlations pool both housing conditions. At least three pairwise-complete observations and variation in both variables are required.
- Four of the 32 relationships have no fixed literature direction. Conditional directional recovery uses the estimable targets among the remaining 28; unconditional recovery uses all 28. Knowledge-probe literature agreement also uses those 28.
- Bootstrap directional agreement is the percentage of estimable resamples matching the fixed literature direction. Estimability uses all 1,000 resamples. Draws preserve the 16 original stratum sizes and keep both housing outputs paired; the same profile draws are used across models and retained prompt versions. Mixed expectations remain unscored. This analysis does not calculate confidence intervals.
- Behavioral prompt sensitivity compares the signs of within-profile housing differences across 224 profiles and nine nonbinary outcomes: 2,016 comparisons per model. Zero/zero differences count as agreement. Knowledge-prompt agreement compares exact increase/decrease/mixed/uncertain categories across all 32 relationships.
- Invariance is assessed across profiles within each model, prompt version, and housing condition. It requires at least two nonmissing observations and one distinct value. The active audit contains 132 checks for eleven outcomes. Paired diagnostics distinguish identical outcomes from opposing differences that cancel.
- CNEP+ MAD compares ten correlations among five shared variables. Each selected profile retains both housing outputs and its matched simulated follow-up record. Confidence intervals use 1,000 original-stratum paired bootstrap resamples with seed 42 and the 2.5th/97.5th percentiles. Required undefined correlations make a resample undefined and withhold its interval.
- CNEP+ follow-up is the nearest eligible post-activation record to baseline plus 0.5 years, without a maximum deviation. `infectiousacute` is normalized to `acute`; eligible states are susceptible, exposed, acute, chronic, and recovered. Timing sensitivity filters these already selected records by the stated tolerance; it calculates no confidence intervals.
- NYC HANES compares twelve covariate correlations with antibody status and generated scores. AUC assesses score rankings. Resampling preserves 35 antibody-positive and 35 antibody-negative participants, keeps all twelve prespecified covariates, and uses 1,000 percentile resamples with seed 42. Survey weights are not applied. An undefined required covariate correlation does not silently reduce the MAD comparison set.

## Outputs and figure terminology

The active quantitative outputs use `cnep_mad.csv`, `cnep_mad_pooled.csv`, `cnep_followup_audit.csv`, `cnep_mad_tolerance_sweep.csv`, `nyc_hanes_mad.csv`, and `nyc_hanes_auc.csv`. Derived CNEP+ correlation columns use `CNEP_`; the NYC HANES MAD estimate is `NYC_HANES_MAD`.

Other outputs include `housing_effects.csv`, `pooled_outcome_correlations.csv`, the literature audit and summary, `prompt_semantic.csv` and `prompt_summary.csv` for housing-difference sign agreement, the two knowledge-prompt comparison CSVs, `invariance_audit.csv`, and the bootstrap summary and 192,000 replicate records. `table_s2.csv` is behavioral prompt agreement; `table_s3.csv` is knowledge-probe direction discrepancies. Table 3 uses the manuscript's metric labels.

Fig2 and FigS1 show bootstrap directional agreement below the observed estimates; knowledge-probe estimates and intervals remain LLM-reported responses. FigS2 shows housing-specific correlations with matched CNEP+ references. FigS3 shows timing sensitivity and retained counts. Captions are in `figures/figure_captions.txt`.

The established dimensions and typography are retained: Fig2 is 160 x 203.6939 mm; both FigS1 pages are 174 x 207.4333 mm. Fig2/FigS1 use bundled, licensed Liberation Sans fonts. Model arrows are 9.4 pt, expected arrows 13 pt, estimates 8.2 pt, and knowledge estimates/agreement lines 7.2 pt. PDF page boxes are fixed without tight cropping. PNG/TIFF exports are RGB at 600 dpi, except FigS3 at 1200 dpi. FigS1 has separate page1/page2 exports and no duplicate first-page alias. Fonts and their license remain active dependencies.

## Local archives and audit records

Local archives preserve exploratory work and earlier source versions. These folders and machine-specific audit logs are excluded from GitHub. They are not needed by the active workflow.

The active workflow excludes unreported prompt-change magnitude intervals, categorical-neighborhood invariance, lifetime-HCV summaries, token-count diagnostics, and correlations outside the prespecified targets. It uses the audited code and paper results listed above.

The runner creates fresh execution logs and input-hash checks when it is run. Bootstrap replicate records are also regenerated locally; the bootstrap summary needed by the figures is included in the repository.
