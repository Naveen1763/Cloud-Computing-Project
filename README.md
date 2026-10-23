# Conformal Prediction for Uncertainty-Aware Serverless Autoscaling

An evaluation and extension of **AAPA** (Archetype-Aware Predictive Autoscaler,
Zhang et al., IEEE HPEC 2025).

This project reproduces AAPA, identifies three defects in its published
evaluation, and replaces its uncertainty mechanism with **split conformal
prediction** — producing a set of plausible workload archetypes with a
distribution-free coverage guarantee instead of a single confidence score.

**Headline result:** 40% fewer SLO violations than AAPA at lower total compute,
on 39,212 scaling decisions derived from real Azure Functions traces
(p = 3.4 × 10⁻⁷, McNemar's paired test). An ablation control confirms the
prediction set — not any incidental simplification — is responsible.

---

## The idea in one paragraph

Autoscaling decides how many pod replicas serve an application. Too few and
requests breach the latency target; too many and compute is wasted. AAPA
classifies each workload into one of four behavioural archetypes — SPIKE,
PERIODIC, RAMP, STATIONARY — and applies a policy tuned to that archetype,
inflating capacity by a margin derived from a single calibrated confidence
value.

A single number can say *how* uncertain the classifier is, but not *what it is
uncertain between*. If the runner-up archetype is SPIKE, the required capacity
is 2.5× that of PERIODIC; a uniform margin cannot distinguish the two cases.
Conformal prediction supplies the missing information as a **set** of archetypes
that cannot be statistically ruled out, letting the controller provision for the
alternatives that actually remain.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
```

### View the results

```bash
streamlit run app.py
```

Opens a dashboard at `http://localhost:8501` presenting every saved
experiment. It reads the result files directly and recomputes nothing, so it
loads instantly.

| Page | Shows |
|---|---|
| **Overview** | Headline metrics, failures chart, full results table |
| **Controller comparison** | Failures and cost/reliability charts, clean vs ambiguous split, per-archetype breakdown |
| **Evaluation findings** | The three defects found in the original evaluation, each with before/after |
| **Conformal analysis** | Coverage, set-size distribution, per-class coverage, MAPIE validation |
| **Limitation** | The capacity ladder and why set-based hedging stalls on SPIKE |

The sidebar switches between four saved runs, so the effect of the train/test
split and the load regime can be compared side by side.

For a terminal summary instead:

```bash
python examples/demo_results.py --results-dir results_loadfix
```

---

## Results

Measured on 39,212 scaling decisions. Each controller sees the same window and
is judged against the same next-window demand, so differences are attributable
to decision logic alone.

| Controller | SLO violations | Mean pods | vs AAPA |
|---|---|---|---|
| Reactive (Kubernetes HPA) | 183 | 227.9 | +135% violations |
| Ablation control | 90 | 507.6 | +15% violations |
| **AAPA (paper)** | **78** | **539.4** | baseline |
| **Conformal sets (ours)** | **47** | **531.3** | **−40% violations** |
| Conformal + warm pool | 47 | 534.3 | −40% violations |

### Statistical validation

McNemar's paired test, valid because every controller is scored on identical
windows:

| Comparison | Discordant pairs | p |
|---|---|---|
| Ours vs AAPA | 35 AAPA-only failures, 4 ours-only | 3.4 × 10⁻⁷ |
| **Ours vs ablation control** | **43 vs 0** | 2.3 × 10⁻¹³ |
| Ours vs HPA | 136 vs 0 | 2.3 × 10⁻⁴¹ |

The control comparison is **43–0** — our method never fails on a window the
control handles. That is strict dominance, and it is the evidence that the
prediction set is doing the work.

### On the compute figure

Total compute is lower, but the effect is concentrated:

| Window size | Count | Our cost vs AAPA |
|---|---|---|
| Needs 1 pod | 27,173 | +1.0 pod |
| Needs 2–10 pods | 8,661 | +2.0 pods |
| Needs 11–100 pods | 1,380 | +27 pods |
| Needs >100 pods | 1,998 | **−199 pods** |

We are *more* expensive on 95.8% of windows. The aggregate saving comes from
large windows, where AAPA's multiplicative margin scales with the provisioning
base while our hedging is bounded by the archetype capacity ladder. Since
cloud cost is the sum rather than the median, total compute is the meaningful
measure — but the claim is about totals, not about every decision.

---

## Defects found in the original evaluation

### 1. The train/test split leaks workloads

The published pipeline splits by day: days 1–9 of a workload train the model,
days 12–13 of the *same* workload test it. A workload's archetype never
changes, so the classifier can score well by recognising the workload and
recalling its label. Verified: **100% of test workloads also appear in
training**.

| | Split by day (original) | Split by workload (fixed) |
|---|---|---|
| Classifier accuracy | 96.4% | 90.7% |
| Predictions naming one archetype | **95.2%** | **1.4%** |

Accuracy barely moves, but apparent certainty collapses. The confidence was an
artefact of the leak — and while 95% of predictions name a single archetype,
any set-based method has nothing to act on.

### 2. The weak-supervision labels are circular

Labels are computed from the same window statistics given to the classifier as
features, so training against them measures how well the model reproduces the
labelling functions rather than how well it recovers workload behaviour. They
agree with the dataset's own ground-truth archetypes on only **18.6%** of
windows, which explains the reported 99.8% accuracy.

### 3. SPIKE is deleted before evaluation

The windowing stage drops any workload active in fewer than 10% of minutes.
Bursty traffic is idle by definition, so the filter removes **42 of 43 SPIKE
workloads (98%)** — the archetype on which the paper's strongest claims rest.

---

## Known limitations

**The capacity ceiling.** Archetypes form a ladder of capacity requirements:

```
PERIODIC 1.33×  <  RAMP 1.67×  <  STATIONARY 1.82×  <  SPIKE 3.33×
```

Taking the maximum over the prediction set adds capacity only when the set
contains something more demanding than the top-ranked prediction. When SPIKE is
predicted it already occupies the highest rung, so the rule returns the unhedged
answer and the method degenerates to the control (measured: 403 vs 405 failures
with SPIKE present). **The headline result therefore holds specifically where
the top-ranked archetype is not the most resource-intensive one.**

**Coverage is not uniform.** PERIODIC coverage measures 0.739 against a 0.90
target, attributable to distribution drift between calibration and test. Overall
coverage holds at 0.939 because STATIONARY dominates.

**Single-seed runs.** Results are from one split. The workload partition is the
dominant variance source, so repeated trials with confidence intervals are
needed before these numbers are treated as final.

---

## Reproducing the experiments

The pipeline runs on the bundled KSWD traces. **Note:** `dataset/workloads/`
(8.7 GB) is excluded from this repository — see *Data* below.

```bash
# 1. Train the classifier and calibrate the conformal predictor
python examples/run_classification.py --max-workloads 200 --labels kswd \
    --split workload --mondrian --output-dir models_gt

# 2. Decision-level controller comparison  (produces the headline result)
python examples/run_conformal_controllers.py --max-workloads 200 --mondrian \
    --split workload --load-multiplier 60 --results-dir results_loadfix

# 3. Full discrete-event simulation
python examples/run_simulation.py --model-dir models_gt \
    --max-workloads 24 --sim-minutes 360

# 4. Validate the conformal implementation against MAPIE
python examples/validate_conformal_mapie.py --model-dir models_gt

# 5. Sweep alpha / score function without retraining
python examples/run_conformal_evaluation.py --model-dir models_gt --plot

# 6. Regenerate report figures
python examples/make_slides.py --results-dir results_loadfix --mono \
    --out-dir report_figs
```

### Key flags

| Flag | Effect |
|---|---|
| `--split workload` | Leak-free split. `day` reproduces the paper's protocol. |
| `--labels kswd` | Ground-truth archetypes. `weak` uses the paper's labelling functions. |
| `--mondrian` | Class-conditional calibration; protects rare archetypes. |
| `--load-multiplier` | Scales trace load. The traces are mostly idle, so 1× leaves 95% of windows needing a single pod. |
| `--min-activity 0.0` | Disables the filter that removes SPIKE workloads. |

---

## Project structure

```
app.py                          Streamlit results dashboard
dashboard/
  data.py                       Loading, aggregation, recorded findings
  charts.py                     Plotly figures
src/
  core/
    simulator.py                SimPy Kubernetes simulator + all autoscalers
    strategies.py               Archetype-specific scaling policies
    autoscaler.py               AAPA controller
  data/
    kswd_loader.py              Bundled dataset loader              [added]
    loader.py                   Azure Functions CSV loader
    windowing.py                Sliding windows, train/test splits
    features.py                 37-feature extraction
  models/
    conformal.py                Split conformal prediction          [added]
    calibration.py              Beta calibration baseline           [added]
    classifier.py               LightGBM archetype classifier
    labeling.py                 Weak supervision
    uncertainty.py              Conformal scaling policy
  utils/
    rei.py                      Resource Efficiency Index           [added]
    metrics.py                  Evaluation metrics
examples/                       Runnable pipeline scripts
results_*/                      Saved experiment outputs
AAPA_Conformal_Report.docx      Full project report
OPTIMIZATION_ROADMAP.md         Findings and prioritised next steps
```

`[added]` marks modules written for this project. The remainder is the AAPA
codebase, repaired — the published version did not execute (broken imports, and
example scripts calling an API absent from the codebase).

---

## Methods implemented

**Classification** — LightGBM over four archetypes, 37 statistical features from
60-minute sliding windows with a 10-minute stride.

**Baseline uncertainty** — beta calibration (Kull et al., 2017), the mechanism
the AAPA paper specifies. It was *not* present in the published code, which used
raw softmax against hardcoded thresholds, so it was implemented here to provide
an honest baseline.

**Our mechanism** — split conformal prediction with LAC and APS nonconformity
scores, marginal and Mondrian class-conditional calibration. Cross-validated
against MAPIE: LAC agrees set-for-set at every α; randomised APS agrees on
coverage and mean set size.

**Controllers compared** — Reactive/HPA, AAPA with beta-calibrated margin, two
conformal variants, and a no-hedge ablation control.

---

## Data

`dataset/workloads/` holds 10,000 workloads with full 14-day per-minute traces
(8.7 GB) and is **excluded from this repository** by `.gitignore`. The dataset
schema, archetype groupings, tools and examples are included.

To obtain the traces, either:

- download the KSWD dataset from the
  [original AAPA repository](https://github.com/GuilinDev/aapa-simulator), or
- regenerate from the Azure Functions 2019 dataset using
  `dataset/tools/azure_to_k8s_converter.py`.

The saved results in `results_*/` are included, so **the dashboard works on a
fresh clone without the dataset**.

---

## Future work

- **Predict capacity directly rather than archetype.** Conformalized quantile
  regression on next-window demand gives a calibrated upper bound with no
  ceiling, and the guarantee `P(demand ≤ provisioned) ≥ 1 − α` *is* the SLO
  promise rather than a proxy for it.
- **Conformal risk control.** Calibrate against expected capacity shortfall
  instead of classification coverage, admitting asymmetric costs for over- and
  under-provisioning.
- **Cost-asymmetric hedging.** Replace the maximum over the set with
  expected-cost minimisation.
- **Repair the data pipeline**, in particular the activity filter that removes
  SPIKE.
- **Repeated trials with confidence intervals.**
- **Validation on a live Kubernetes cluster.**

---

## References

1. Zhang, G. et al. *AAPA: An Archetype-Aware Predictive Autoscaler with
   Uncertainty Quantification for Serverless Workloads on Kubernetes.* IEEE
   HPEC, 2025.
2. Shahrad, M. et al. *Serverless in the Wild.* USENIX ATC, 2020.
3. Kull, M., Silva Filho, T., Flach, P. *Beta Calibration.* AISTATS, 2017.
4. Vovk, V., Gammerman, A., Shafer, G. *Algorithmic Learning in a Random
   World.* Springer, 2005.
5. Romano, Y., Sesia, M., Candès, E. *Classification with Valid and Adaptive
   Coverage.* NeurIPS, 2020.
6. Angelopoulos, A., Bates, S. *A Gentle Introduction to Conformal Prediction.*
   2021.
7. Angelopoulos, A. et al. *Conformal Risk Control.* 2022.
8. Taquet, V. et al. *MAPIE: An Open-Source Library for Distribution-Free
   Uncertainty Quantification.* 2022.

---

## License

MIT — see [LICENSE](LICENSE). The underlying AAPA simulator is MIT licensed by
its original authors.
