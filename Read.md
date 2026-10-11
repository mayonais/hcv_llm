# HCV LLM evaluation

This repository evaluates LLM-generated behavioral outputs for people who inject drugs against literature-informed association directions, agent-based model (ABM) correlations, and empirical HCV survey associations. The evaluated models are Claude Sonnet 4.6, DeepSeek v3.2, and GPT-OSS-120B.

## Notebooks and shared code

| File | Purpose |
| --- | --- |
| `llm_prompts.ipynb` | Select the 224 synthetic profiles and construct housing-scenario and knowledge-probe prompts. Optional model generation is disabled by default. |
| `llm_outputs_analysis.ipynb` | Analyze knowledge agreement, housing effects, generated-outcome correlations, ABM correlation MAD, observation-timing sensitivity, and prompt sensitivity. |
| `icpsr.ipynb` | Construct NYC HANES profiles and evaluate saved HCV risk scores using survey correlation MAD and AUC. |
| `tables.ipynb` | Generate the housing and correlation figures and the Table 3 summary from computed results. |
| `analysis_utils.py` | Shared target definitions, rank statistics, matched-agent bootstrap, scoring, and table formatting. |
| `association_targets.json` | The 32 retained variable pairs and their prespecified directions. |
| `run_analysis.py` | Execute the notebooks from saved inputs and validate the results, without enabling model generation. |

`abm_analysis.ipynb` is an earlier exploratory notebook. The four notebooks listed above form the current workflow.

## Setup and inputs

Install the Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Run all commands and notebooks from the repository root. The saved-response inputs are:

- `outputs/outputs.csv`: generated profile outcomes, indexed by model, agent, housing condition, and prompt version.
- `outputs/knowledge_outputs.csv`: saved knowledge-probe responses. Historical records are retained; analysis selects the 32 current relationships and excludes the fentanyl question.
- `outputs/outputs_nyc_hanes.csv`: survey profile characteristics, observed HCV antibody status, and generated risk scores.
- `2018-09-13--06.02.24.events.csv`: ABM event records; the notebooks read this file with `header=4`.

For the complete survey profile-construction workflow, obtain the ICPSR 31421 Stata files and retain this directory structure:

```text
ICPSR_31421/
  DS0001/31421-0001-Data.dta
  DS0002/31421-0002-Data.dta
  DS0003/31421-0003-Data.dta
  DS0004/31421-0004-Data.dta
  DS0005/31421-0005-Data.dta
  DS0006/31421-0006-Data.dta
```

These local survey downloads are not included in this repository. Checking the saved results or regenerating the tables and figures does not require the Stata files.

## Run the analysis

Execute the complete workflow after placing the required survey files:

```sh
python run_analysis.py
```

Alternatively, run `llm_prompts.ipynb`, `llm_outputs_analysis.ipynb`, `icpsr.ipynb`, and then `tables.ipynb`. To regenerate only the tables and figures from the included result CSVs, run `tables.ipynb`.

Validate the included results without executing the notebooks:

```sh
python run_analysis.py --check
```

The runner checks the directional-scoring counts, statistical helpers, bootstrap completeness, and ABM follow-up selection. Comparisons with archived local versions are performed only when an archive is present.

## Evaluation definitions

- There are 32 retained knowledge-probe relationships. Four have no fixed directional benchmark, leaving **28 directional targets**. Conditional recovery uses the number of estimable targets; unconditional recovery uses all 28.
- Housing effects use paired Cohen's `d_z`: the mean homeless-minus-stable difference divided by the standard deviation of the paired differences.
- Generated-outcome Spearman correlations combine original-prompt rows from both housing conditions.
- Main ABM MAD combines both housing-condition rows, aligns each with the same agent's ABM snapshot, and resamples whole agents with both conditions retained. Condition-specific correlations are displayed separately.
- ABM `infectiousacute` labels are normalized to `acute`; vaccinated records are excluded. The closest eligible post-activation record to the six-month target is selected. Timing sensitivity restricts these selected records to narrower windows and has no bootstrap intervals.
- Survey MAD compares covariate correlations with observed HCV antibody status and with generated risk scores. Survey bootstrap sampling preserves the positive and negative group counts.
- Intervals use 1,000 percentile bootstrap resamples with seed 42. Housing-direction sensitivity compares stored `Paraphrase=0` with `Paraphrase=2` across nine numeric outcomes; zero differences under both prompts count as agreement. Knowledge-prompt stability compares exact direction categories across all 32 retained relationships.

## Results and figures

`analysis_results/` contains the computed CSV summaries and audits, including `table3.csv`, `table_s3.csv`, `literature_target_audit.csv`, `abm_mad_pooled.csv`, `empirical_mad.csv`, and `empirical_auc.csv`. `verification_manifest.json` records the verification environment and input, code, and result hashes.

`figures/` contains the housing comparison (`housing_effect`), generated-outcome correlation comparison (`main_corr`), condition-specific ABM comparison (`abm_corr2`), and observation-timing sensitivity (`abm_mad_tolerance_sweep`).

## Optional model generation

Generation requires `boto3` and a configured AWS Bedrock account with access to the specified model IDs. `RUN_GENERATION=False` is the default. Generation writes to `generated_outputs/`; analyzing new generations requires explicitly selecting that directory through `DATA_DIR` in `analysis_utils.py` and the empirical input path in `icpsr.ipynb`. The default workflow uses the included saved responses.
