# Optimization roadmap

Where the conformal autoscaling work stands, and what to do next.

## The diagnosis driving all of this

Measured on the leak-free workload split (39,212 decision windows,
`results_workload_split/`):

- **99.82%** of windows are already served correctly with no hedging at all
- Of the +2.25 pods/window our method spends over the no-hedge control,
  **98.7% lands on windows that never needed it**
- We rescue **32 of 70** failures the control suffers

So hedging pays on roughly 0.2% of windows but is applied to all of them.
The problem is not set size so much as **how often we act on the set**.

## Current position

| Method | Failures / 39,212 | Pods/window |
|---|---|---|
| Reactive (HPA-style) | 153 | 5.70 |
| Top1_NoMargin (control) | 70 | 10.39 |
| AAPA_Beta (paper) | 57 | 11.97 |
| **Conformal_TargetOnly (ours)** | **38** | **12.64** |
| Conformal_Envelope (ours) | 34 | 15.64 |

Classifier: 90.7% accuracy, avg set size 2.02, 98.6% of windows ambiguous.

---

## Tier 1 - highest impact

### 1. Risk-gated hedging
Only hedge when under-provisioning is actually plausible. Compute the top-1
sizing, compare against plausible next-window demand, and skip the hedge when
headroom is already comfortable. Directly attacks the 98.7% waste.

### 2. Conformal Risk Control
Replace the coverage guarantee ("true archetype is in the set 90% of the
time") with a loss guarantee ("expected capacity shortfall <= epsilon").
Coverage is only a proxy for what we care about; risk control targets the real
quantity, and generalises conformal prediction to any monotone loss.

- Angelopoulos et al., *Conformal Risk Control* -
  https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/conformal-risk.pdf
- Bates et al., *Distribution-free Risk-Controlling Prediction Sets* -
  https://dl.acm.org/doi/fullHtml/10.1145/3478535

### 3. Cost-asymmetric hedging instead of hard `max`
A set of {PERIODIC p=0.85, STATIONARY p=0.15} currently pays full STATIONARY
capacity for a 15% chance. Minimise expected cost with an explicit price on a
broken promise instead.

- *Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control* -
  https://arxiv.org/pdf/2302.00839
- *Optimal Decision-Making Based on Prediction Sets* -
  https://arxiv.org/pdf/2602.00989

## Tier 2 - smaller sets

### 4. RAPS instead of APS
Regularized Adaptive Prediction Sets penalises low-ranked classes, giving
smaller and more stable sets at equal coverage. Hyperparameters `k_reg`,
`lambda`. MAPIE implements it.
- https://mapie.readthedocs.io/en/latest/theoretical_description_classification.html

### 5. Improve the classifier
Sets are wide because accuracy is 90.7%, and the confusion is almost entirely
PERIODIC <-> STATIONARY. Stronger periodicity features (autocorrelation peak
prominence, Lomb-Scargle) would shrink sets at the source. A better classifier
needs less hedging - the cheapest cost reduction available.

### 6. Per-archetype alpha by cost
SPIKE failures are expensive; PERIODIC failures are nearly free (0.0001 either
way). Demand 99% coverage for SPIKE, 80% for PERIODIC. The Mondrian code
already computes per-class thresholds, so this is a small extension.
- *Cost-Sensitive Conformal Prediction* - https://arxiv.org/html/2607.27143

## Other fixes

### 7. Load regime  [DONE - `--load-multiplier`]
99.82% of windows succeed with no hedging, so autoscaling barely matters at
this load. Raise request rate or lower pod capacity until the baselines
genuinely struggle. May change the conclusions more than anything in Tier 1.

### 8. Multiple seeds
Everything is n=1. The workload split is now the dominant variance source.
The paper uses 5 trials with 95% CIs.

### 9. Drift / online recalibration
Coverage decays 0.925 -> 0.845 under the temporal split. Adaptive conformal
inference addresses exactly this.
- *Test-time Recalibration under Distribution Shift* -
  https://arxiv.org/pdf/2210.04166

### 10. CPU target inconsistency  [DONE - 0.55 everywhere]
STATIONARY is 55% in the paper's Table III and in `src/core/strategies.py`,
but 65% in `src/core/simulator.py`. Pick one.

---

## NEW - diagnosis of where the pods actually go

Measured on `results_loadfix/` (60x load, workload split):

    mean provisioned : 531.3
    mean required    : 159.2
    mean excess      : 372.1

Decomposition of the 372-pod excess:

| Source | Pods | Note |
|---|---|---|
| CPU-target divisor | **371.8** | 99.9% of the waste |
| Window-to-window forecast error | 0.3 | current window predicts the next almost exactly |

**Better forecasting is a dead end.** The provisioned/required ratio is 3.34,
which is exactly 1/0.30 - SPIKE's CPU target. Because hedging takes `max` over
the set and SPIKE has the lowest target, any set containing SPIKE is sized at
3.33x. We pay spike-grade headroom on essentially every window.

### 11. SPIKE is structurally deleted from the dataset  [CRITICAL]
`SlidingWindowProcessor.process_all_functions` defaults to
`min_activity_threshold=0.1`, dropping any workload active less than 10% of
the time. Spiky traffic is by definition mostly idle, so:

| Archetype | Dropped |
|---|---|
| SPIKE | 42 of 43 (98%) |
| PERIODIC | 52 of 66 (79%) |
| STATIONARY_NOISY | 1 of 66 (1.5%) |

The test set therefore contains **no SPIKE workloads at all** - we hedge at
3.33x against an archetype that never occurs. This also undermines the paper's
SPIKE-specific claims, since its own pipeline would discard those traces.
Fix: expose the threshold and lower it.

### 12. PERIODIC under-covers badly
Coverage 0.739 against a 0.90 target, with Mondrian enabled. The guarantee is
being violated for that class; likely the same drift effect measured earlier.
Must be fixed - it undermines the central claim.

### 13. Revisit SPIKE's 30% CPU target
It comes from the paper's Table III but is the single largest cost driver in
the system (3.33x multiplier). A sensitivity analysis at 40% / 50% is a
legitimate finding about AAPA's parameters.

## THE STRUCTURAL CEILING - why set-based hedging stalls on SPIKE

Measured on `results_v2/` (300x load, SPIKE restored via `--min-activity 0`,
84,313 windows):

| Controller | Failures | Pods |
|---|---|---|
| Reactive | 585 | 12.73 |
| Top1_NoMargin (control) | 405 | 18.31 |
| **Conformal_TargetOnly (ours)** | **403** | 18.80 |
| AAPA_Beta (paper) | 201 | 19.63 |
| Conformal_Envelope | 194 | 20.32 |

Archetypes imply a capacity ladder (pods per unit of traffic):

    PERIODIC 1.33x  <  RAMP 1.67x  <  STATIONARY 1.82x  <  SPIKE 3.33x

`max` over the prediction set only adds capacity when the set contains
something **more demanding than the top-1 guess**. When the model predicts
SPIKE, SPIKE is already the top rung - `max` returns the top-1 answer and the
method degenerates to the no-hedge control. Hence 403 vs 405 failures.

AAPA's margin has no such ceiling: it multiplies *on top of* the target
(3.33x * (1 + 0.5(1-c))), so it keeps adding capacity exactly where it is
needed. `Conformal_Envelope` survives for the same reason - its warm pool is
additive rather than a max over the ladder.

**Consequence:** set-based hedging can redistribute capacity among archetypes
but can never exceed the most aggressive one. The earlier 40% win was real but
scope-limited - it was measured on data from which the activity filter had
removed 98% of SPIKE workloads, so the top-1 guess was usually *not* the top
rung.

## FUTURE DIRECTION - move the guarantee off the archetype

The method answers "which archetype is this?" when the question that decides
success is "how much capacity do I need?". Those coincide only while the
archetype -> capacity mapping has headroom left.

### A. Conformal prediction on demand  [RECOMMENDED]
Drop archetype classification from the control path. Predict next-window peak
demand directly with a calibrated upper bound:

    "next-window peak <= U with 90% confidence"  ->  provision U

Conformalized quantile regression (CQR). Why it is the better target:

1. **No ceiling** - the bound rises as far as the data requires.
2. **The guarantee is the goal**, not a proxy: P(demand <= provisioned) >= 1-alpha
   *is* the SLO promise.
3. **Bypasses everything broken upstream** - circular weak labels, the leaky
   day split, and the activity filter that deletes SPIKE. No archetype labels
   needed at all.

The existing infrastructure (calibration splits, coverage evaluation, the
controller harness) carries over largely unchanged.

### B. Conformal Risk Control  [intermediate]
Keep the archetype pipeline but calibrate a capacity *multiplier* against
measured shortfall rather than hedging between classes. Also escapes the
ceiling; smaller rewrite than A.

### Narrative for a write-up
"Set-based archetype hedging has a structural ceiling at the most aggressive
archetype. We demonstrate this empirically, then move the conformal guarantee
onto the quantity that actually determines SLO satisfaction."

## Findings already established

1. **Weak labels are circular** - they agree with KSWD ground-truth archetypes
   only 18.6% of the time, which is why accuracy reads 99.8%.
2. **The day-based split leaks** - the same workload appears in train and test,
   and archetype is a per-workload constant, so the classifier memorises rather
   than generalises. Fixing it drops the singleton rate from 95% to 1.4%.
3. **AAPA's beta-calibrated margin is poor value** - on the leaked split it cost
   21% more compute for zero SLO benefit.
4. **SPIKE under-covers** at 0.71-0.85 against a 0.90 target under marginal
   calibration; Mondrian calibration is required.
5. **Conformal beats beta calibration once the split is honest** - 33% fewer
   violations for 5.6% more compute, and the no-hedge control loses, confirming
   the prediction set is doing the work.
6. **Implementation validated against MAPIE** - LAC agrees set-for-set.
