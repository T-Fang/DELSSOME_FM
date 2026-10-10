# DELSSOME-FM v2 — architecture

**Scope.** Parcellation fixed at $N = 68$ (Desikan–Killiany). The research claim is model-agnosticism alone: one surrogate that transfers across circuit models without retraining from scratch. Parcellation-agnosticism is deferred, not withdrawn — v1's signed-graph region encoder remains the design for that, and nothing here forecloses it.

**Companion documents.** `docs/generation.md` (what the synthetic corpus contains, from `delssome-fm-poc-generation.md`); `docs/data.md` (the empirical data and the comparison with the original DELSSOME); `delssome-fm-v2-synthetic-model-generation.md` (fuller model survey and later-stage generation rules, not in this repository).

**Status (2026-10-10).** Nothing in this document is implemented yet (build step 8). Where the corpus being generated fixes a quantity used here, the value is noted.

---

## 1. Symbols and shapes

| Symbol | Shape | Meaning |
|---|---|---|
| $N$ | 68 | regions, fixed |
| $V$ | 1–6 | state variables per region, varies by model (corpus: 95% of valid models have $V \le 2$) |
| $P$ | 1–5 | regional free parameters, varies by model (corpus: §9 question 5) |
| $K$ | 1–2 | global free parameters, varies by model |
| $M$ | ~20–180 | nodes in the equation DAG, varies by model (corpus: median 36, 5–95% range 19–67, max 177) |
| $d$ | 128 | transformer width |
| $d_t$ | 128 | template encoder width |
| $k$ | 32 | bilinear FC factorization rank |
| $\Theta$ | $N \times P$ | regional parameter fields |
| $\Psi$ | $K$ | global parameters |
| $C$ | $N \times N$ | structural connectivity |
| $X_{\text{dag}}$ | $M \times 21$ | DAG node features: 15 type one-hot, **2 constant** (sign, $\log_{10}\lvert c\rvert$), 1 parameter index, 1 scope tag, 2 observation |
| $\text{FC}^{\text{emp}}$ | $N \times N$ | empirical functional connectivity |
| $\text{FCD}^{\text{emp}}$ | 100 | empirical FCD, as a CDF at 100 fixed levels |

Indices: $i, j$ over regions; $p$ over regional parameters; $\kappa$ over globals; $v$ over state variables; $m$ over DAG nodes.

---

## 2. Template encoder

A Graphormer over the equation DAG produces per-node embeddings.

$$H_{\text{tmpl}} = \text{Graphormer}(X_{\text{dag}}, A_{\text{dag}}) \;\in\; \mathbb{R}^{M \times d_t}$$

Four readouts are extracted by node role:

| Readout | Shape | Extracted from |
|---|---|---|
| $\kappa_p$ | $P \times d$ | the `Par` node for the $p$-th **regional** free parameter |
| $\lambda_\kappa$ | $K \times d$ | the `Par` node for the $\kappa$-th **global** free parameter |
| $u$ | $d$ | mean over the $V$ `Deriv` root nodes |
| $o$ | $d$ | the single `Out` node |
| $m_{\text{agg}}$ | $d$ | mean over the `Agg` nodes (1 or 2); zero if the model has none |
| $z_M$ | $d$ | mean over all $M$ nodes, a graph-level summary |

$\kappa_p$ is what makes the parameter encoder model-agnostic: it tags each parameter by *where in the equation it sits*, so the network learns "this is the coefficient multiplying a state inside a transfer function" rather than "this is column 2."

### Constant encoding

Two columns, nonzero on `Const` nodes only:

$$\text{col 1} = \operatorname{sign}(c) \in \{-1, 0, +1\}, \qquad \text{col 2} = \operatorname{clip}\big(\log_{10}\lvert c\rvert,\; -4,\; 4\big)$$

This replaces the raw-value column. Constants are sampled log-uniformly over $[10^{-4}, 10^{4}]$, so two `Const` nodes in one model can differ by a factor of $10^8$, and a raw value would hand that difference straight to the first linear layer.

Column 2 is a **linear map of the sampling variable**: $\log_{10}\lvert c\rvert$ is uniform on $[-4, 4]$ by construction, so there is no compression, no skew, and no region of the range with worse resolution than another. An earlier draft proposed three columns, `[sign, log(1+|c|), c/(1+|c|)]`, which is the generic recipe for encoding a real of unknown distribution. The third column exists only to repair the resolution $\log(1+\lvert c\rvert)$ destroys below $\lvert c\rvert = 1$, which is half the prior. Knowing the prior removes the need for the repair.

Sign is a separate column rather than folded in, because it is structural: a positive $L_{vv}$ is a leak and a negative one is growth, which is the difference between MFM and Hopf. A signed log would be discontinuous at $\lvert c\rvert = 1$ and would place $c = 10$ and $c = 10^{-1}$ adjacent in the encoding despite being four decades apart.

Two details. The **clip** matters for hand-written cards, whose constants need not respect the sampler's prior. And $\operatorname{sign}(0) = 0$ is a safety net: a zero constant should never reach the encoder, since masked entries produce no term and canonicalisation removes $0 \cdot x$, so encoding it as $(0, \cdot)$ rather than $(\cdot, -\infty)$ makes a compiler bug surface as a diagnostic rather than a NaN.

Column 2 is roughly 5× the standard deviation of the one-hot columns, so the encoder's input LayerNorm is doing real work here rather than being incidental.

### The output node

The DAG has two kinds of root: $V$ `Deriv` roots, one per state variable, and one `Out` root carrying the observable. `Out` attaches to an arbitrary subexpression, so it covers both a single state variable (MFM, FIC, Hopf, MPR) and a difference of two (Jansen–Rit's $y_1 - y_2$).

Two node-feature columns carry a one-hot for the observation model (Balloon–Windkessel, direct downsample). They follow the same convention as the constant columns: present on every node, nonzero only on the node where they mean something, here `Out`.

**$o$ needs its own readout rather than being left to $z_M$.** $z_M$ averages over all $M$ nodes, so one node in forty barely registers, but the observation model changes the statistics profoundly. Balloon–Windkessel is a low-pass filter with roughly a 6 second impulse response, erasing everything above about 0.2 Hz. A fast model seen through it loses all its structure above that; the same model downsampled directly does not. Not something to average away.

Putting the observable in the graph also makes the **path from each `Deriv` root to `Out`** visible to the Graphormer's shortest-path attention bias. That is how the encoder learns that FIC observes $S^{(E)}$ and not $S^{(I)}$.

**$m_{\text{agg}}$ is a readout, not a modulator.** It originally existed to FiLM-modulate the SC encoder, and was briefly dropped along with the FiLM (§3) on the grounds that it had no remaining consumer. That inference was too quick: the right question was whether it should have a *different* consumer, not whether its old one had gone.

The `Agg` node embeddings carry the one structural fact nothing else summarises — **where the coupling sum sits in the equation**. Depth 2 for Linear and Hopf, depth 5 to 7 for MFM, FIC and Wilson–Cowan. That is the most discriminating feature across the model survey and precisely what the Graphormer's shortest-path attention bias exists to read. The same embeddings also carry direct versus diffusive, and whether $f_c$ is the identity.

The argument for dropping it ("the `Agg` nodes are still visible as template tokens") proves too much: it would equally delete $u$ and $o$, whose nodes are also in the sequence. The reason $o$ needs its own readout — $z_M$ averages one node in forty into irrelevance — applies verbatim here.

---

## 3. Region tokens

### 3.1 Parameter half

$$t_{i,p} = W^{\text{reg}}\,\Theta[i,p] + \kappa_p \;\in\; \mathbb{R}^{d}, \qquad W^{\text{reg}} \in \mathbb{R}^{d \times 1}$$

$$t^{\text{par}}_i = \text{MLP}_\theta\Big(\tfrac{1}{P}\textstyle\sum_{p} t_{i,p}\Big) \;\in\; \mathbb{R}^{d/2}$$

**This must not be a $P$-width MLP.** Original DELSSOME maps $(w_{EE}, w_{EI}, \sigma)_i \to \mathbb{R}^{d/2}$ with a single layer of input width 3, which hard-codes $P = 3$. Embedding each parameter as a *scalar* and summing with its positional tag $\kappa_p$ is the only construction that is free in both $N$ and $P$, and it costs nothing.

### 3.2 SC half

$$t^{\text{SC}}_i = \text{MLP}_{\text{SC}}\big(C[i,:]\big) \;\in\; \mathbb{R}^{d/2}, \qquad \text{MLP}_{\text{SC}}: \mathbb{R}^{68} \to \mathbb{R}^{d/2}$$

Unmodulated. Two notes on what that means:

**No $G$ scaling.** The paper writes $G \cdot C[i,:]$ because FIC has exactly one global coupling gain. Synthetic models have $K \in \{1,2\}$, so the gain enters through $g$ (§3.3) instead, and $\text{MLP}_{\text{SC}}$ sees the raw connectivity profile.

**No FiLM.** An earlier draft modulated this MLP by $m_{\text{agg}}$ so the SC representation could specialize to how a given model uses coupling. Removed for simplicity. The consequence is that the SC half now encodes *only* "this region's connectivity profile," with nothing about how the model consumes it, and structural information has to reach the transformer by another route.

That route is the template tokens in the sequence, plus $m_{\text{agg}}$ in the global offset $g$. **So the template tokens are now load-bearing rather than an enhancement, and should not be ablated at the same time as FiLM** — doing both would leave $\kappa_p$ and $z_M$ as the only paths, which is thin.

$m_{\text{agg}}$ does something the template tokens cannot: it puts a summary of the coupling structure into **every region token before attention starts**, rather than requiring attention to fetch it. With the FiLM gone, the SC half encodes only the region's connectivity profile and nothing about how the model consumes it, so the two facts have to meet somewhere. This is the cheaper of the two places.

### 3.3 Globals and assembly

$$s_\kappa = W^{\text{glob}}\,\Psi[\kappa] + \lambda_\kappa, \qquad g = \text{MLP}_g\Big(\tfrac{1}{K}\textstyle\sum_\kappa s_\kappa\Big) + z_M + u + o + m_{\text{agg}} \;\in\; \mathbb{R}^{d}$$

$$\boxed{\;\text{token}_i = \big[\,t^{\text{par}}_i \,\|\, t^{\text{SC}}_i\,\big] + g \;\in\; \mathbb{R}^{d}\;}$$

Same mean-then-MLP construction on the global side, for the same reason.

---

## 4. Transformer

The sequence is three segments, each with a learned segment embedding:

| Segment | Length | Content |
|---|---|---|
| CLS | 1 | learned vector |
| regions | 68 | $\text{token}_i$ |
| template | $M$ | $W_{\text{proj}}H_{\text{tmpl}}$, $W_{\text{proj}} \in \mathbb{R}^{d \times d_t}$ |

Total $L = 69 + M \approx 100$–$190$. Standard pre-LN encoder, 4 layers, 8 heads, $d = 128$ — the original DELSSOME configuration.

Template tokens let region tokens attend to individual equation nodes rather than only to pooled readouts. At this sequence length the cost is negligible.

Outputs used downstream: $h_{\text{CLS}} \in \mathbb{R}^{d}$ and $r_i \in \mathbb{R}^{d}$ for $i = 1..68$. Template output positions are discarded.

---

## 5. Stage 1 — pretraining on simulated statistics

### 5.1 Targets

Three groups, all computed from simulated BOLD through **the identical pipeline stage 2 will use**: same TR, same FCD window length and stride, same $\operatorname{arctanh}$.

**Regional** — 6 per region, supervised on $r_i$:

| # | Statistic |
|---|---|
| 1 | BOLD mean |
| 2 | BOLD SD |
| 3 | BOLD skewness |
| 4 | BOLD kurtosis |
| 5 | FC node strength, $\frac{1}{N-1}\sum_{j\neq i}\text{FC}_{ij}$ |
| 6 | lag-1 autocorrelation |

**Pairwise** — supervised on region-token pairs: $\operatorname{arctanh}(\text{FC}_{ij})$, over the 2278 upper-triangular edges.

**Global** — 103, supervised on $h_{\text{CLS}}$: the FCD CDF at 100 fixed FCD values (−0.98, −0.96, …, 1.00; data.md §8), plus the mean, SD, and skewness of the upper-triangular FC.

In the corpus these are the arrays `regional` (68 × 6), `pairwise` (2278) and `fcd_cdf` (100) + `fc_moments` (3) of each simulation, in float32 (generation.md §9). `sim/summary.py::summarize` computes them, and stage 2 must call the same function.

Target vector per simulation: $68 \times 6 + 2278 + 103$.

### 5.2 Heads

**Regional.** $\hat y_i = \text{MLP}_{\text{reg}}(r_i) \in \mathbb{R}^{6}$, weights shared across regions.

**Pairwise — bilinear, and this is the important one.**

$$\phi: \mathbb{R}^{d} \to \mathbb{R}^{k}, \qquad \widehat{\text{FC}}_{ij} = \frac{\phi(r_i)^\top \phi(r_j)}{\sqrt{k}}$$

A single $d$-vector cannot support the FC cost. $1 - r$ is a correlation over 2278 edges against an *arbitrary* empirical FC; computing it from $h_{\text{CLS}}$ alone would require simulated FC to be approximately rank $\sqrt{d}$, which is plausible but unverified, and betting the FC cost on it is a bad trade. The bilinear head is a learned rank-$k$ factorization of simulated FC, trained in stage 1 and consumed by stage 2.

Subsample ~256 edges per training step; materialize all 2278 at evaluation.

**Global.** $h_{\text{CLS}} \to$ two heads: a 3-output MLP for the FC moments, and a 100-output **monotone** head for the FCD CDF, built as a cumulative softmax so the prediction is a valid CDF by construction.

### 5.3 Loss

Every statistic is z-scored across the corpus. Four groups — regional, pairwise, FCD-CDF, FC-moments — combined by uncertainty weighting rather than hand-tuned weights:

$$\mathcal{L} = \sum_{\text{groups } \gamma} \frac{1}{2s_\gamma^2}\mathcal{L}_\gamma + \log s_\gamma$$

with $s_\gamma$ learned. Four scales is manageable to sanity-check by hand, which was part of the reason for cutting the statistic set.

**No task masking is needed.** Every statistic here is universal — any model that produces BOLD produces all of them. The heterogeneous-label machinery from earlier drafts (firing rates for FIC but not Hopf, and so on) is unnecessary at this statistic set. The only exception is a divergent simulation, which has no statistics at all; those rows are excluded from stage-1 loss entirely rather than masked per-statistic. In the corpus they are flagged `diverged` and their statistics are NaN. They make up 14.7% of the draws of valid models.

Draws whose observed signal is flat are also stored, flagged `flat` (0.8% of draws). Their statistics are meaningless: FC is correlation between numerical noise, and about a third of flat draws have some NaN statistics (zero-variance regions). Whether to exclude them too is open (§9 question 6).

### 5.4 Two deliberate redundancies

Regional statistic 5 (FC node strength) and the global FC moments are both derivable from the pairwise targets. This is intentional: they are auxiliary supervision through *different readout pathways*, giving the region token and the CLS a direct low-rank signal about FC rather than requiring them to reconstruct it. Worth knowing it is deliberate, because it does mean FC structure is implicitly upweighted relative to FCD in the total loss.

---

## 6. Stage 2 — empirical cost prediction

### 6.1 Empirical encoders

Reused unchanged from the original DELSSOME: an FC encoder consuming $\text{FC}^{\text{emp}}$, and an FCD encoder consuming the empirical FCD CDF.

$$e_{\text{FC}} = \text{Enc}_{\text{FC}}(\text{FC}^{\text{emp}}), \qquad e_{\text{FCD}} = \text{Enc}_{\text{FCD}}(\text{FCD}^{\text{emp}})$$

### 6.2 Simulated side — both statistics, symmetrically

Both stage-1 predictions are routed through the **same encoders as their empirical counterparts, with shared weights**:

$$e_{\widehat{\text{FC}}} = \text{Enc}_{\text{FC}}(\widehat{\text{FC}}), \qquad e_{\widehat{\text{FCD}}} = \text{Enc}_{\text{FCD}}(\widehat{\text{FCD}})$$

where $\widehat{\text{FC}} \in \mathbb{R}^{68\times68}$ is materialized from the bilinear head and $\widehat{\text{FCD}} \in \mathbb{R}^{100}$ comes from the monotone CDF head. Shared weights put simulated and empirical quantities in one embedding space, which is what makes their difference meaningful to the cost head.

**The two paths are not equally motivated, and it is worth being precise about why.** The region outputs $r_i$ are *not* fed to the cost MLP directly, so $e_{\widehat{\text{FC}}}$ is the only route by which region-level information reaches it — that path carries genuinely new information. $\widehat{\text{FCD}}$, by contrast, is a deterministic function of $h_{\text{CLS}}$, which the cost MLP already receives, so $e_{\widehat{\text{FCD}}}$ adds no information in the strict sense.

It earns its place as **inductive bias rather than information**: encoding the prediction into the same space as the empirical CDF makes their difference directly available, rather than requiring the cost head to re-derive it from a 128-vector. It also stops stage 1 from training a 100-output head that stage 2 discards. One extra encoder call.

### 6.3 Cost heads

$$\text{context} = \big[\,h_{\text{CLS}} \,\|\, e_{\widehat{\text{FC}}} \,\|\, e_{\text{FC}} \,\|\, e_{\widehat{\text{FCD}}} \,\|\, e_{\text{FCD}}\,\big]$$

$$(\hat c_{\text{corr}},\; \hat c_{\text{scale}},\; \hat c_{\text{KS}}) = \text{MLP}_{\text{cost}}(\text{context})$$

targeting the three DELSSOME cost terms: $1 - r(\text{FC}^{\text{sim}}, \text{FC}^{\text{emp}})$, $|\overline{\text{FC}^{\text{sim}}} - \overline{\text{FC}^{\text{emp}}}|$, and $\text{KS}(\text{FCD}^{\text{sim}}, \text{FCD}^{\text{emp}})$.

$h_{\text{CLS}}$ enters the cost MLP directly, alongside the four encodings — it is not routed through any encoder.

### 6.4 The backbone must not be frozen

DELSSOME's advantage over vanilla simulation-based inference comes from training against *empirical discrepancy*. A frozen stage-1 backbone with a cost head bolted on is a simulator emulator — structurally the thing SBI does, and it inherits SBI's exposure to model misspecification rather than DELSSOME's protection from it.

Stage 2 trains: LoRA adapters on the transformer (sweep rank 4/8/16), the full cost heads, and the empirical encoders. The load-bearing claim is *fewer target-model simulations*, not *zero backbone updates*.

### 6.5 Two analytic diagnostics

Both cost terms that compare distributions can also be computed **in closed form** from the stage-1 predictions, with no learned head. Neither is the default — errors in the predictions propagate — but both are cheap and both fail loudly rather than silently.

**KS, and this is the stronger of the two.** $\text{KS} = \max_\ell |\widehat{\text{FCD}}_\ell - \text{FCD}^{\text{emp}}_\ell|$ is a pointwise max over the 100 levels. Given both CDFs you have the KS exactly — nothing is being approximated. If analytic KS tracks the learned head closely, the monotone head is doing real work; if it does not, the stage-1 FCD supervision is not landing, which is a specific and fixable problem rather than a diffuse one.

**FC correlation.** $1 - r(\widehat{\text{FC}}, \text{FC}^{\text{emp}})$ is a correlation over 2278 edges, so it is a more indirect computation and more sensitive to rank-$k$ truncation. If the analytic route is much worse than the learned head, the bilinear prediction is not accurate enough to be carrying the stage-1 backbone — precisely the assumption stage 1 rests on. Sweep $k$ before concluding the approach fails.

---

## 7. Training procedure

| | Stage 1 | Stage 2 |
|---|---|---|
| Data | synthetic corpus, ~100,000 train models × 100 parameter sets (~10M simulations); 10,000 val and 10,000 test models with their own SCs | target-model corpus, deliberately small |
| Input | $\Theta, \Psi, C$, DAG | same, plus $\text{FC}^{\text{emp}}, \text{FCD}^{\text{emp}}$ |
| Target | simulated summary statistics | three cost terms vs. real data |
| Trains | everything | LoRA on backbone, cost heads, empirical encoders |
| SC | **vary across models** | as deployed |

**Vary SC in stage 1.** If every sample uses one group-average connectome, the learned representation entangles with it. Each synthetic model is therefore tied to one of the 64 HCP-YA training group SCs, drawn at random, so SC varies across the corpus without being a separate sampling axis (revised 2026-10-05; generation.md §9). Validation and test group SCs stay unseen in stage-1 training. The val and test synthetic models use them (generation.md §9), so held-out evaluation also tests unseen connectomes.

**Primary metric.** Target-model simulations needed to reach a given optimization regret, pretrained versus from scratch. Secondary: top-$k$ candidate recall under CMA-ES. Cost MSE is diagnostic only — a surrogate with low MSE that misorders the top candidates is useless to the optimizer.

---

## 8. Decisions taken and reversed

Logged so they are not relitigated.

| Decision | Status | Reason |
|---|---|---|
| Signed-graph message-passing region encoder (v1 §6.1) | **deferred** | Needed only for parcellation-agnosticism. Untested, and carries an over-smoothing risk. |
| $P$-width parameter MLP, as in the paper | **rejected** | Hard-codes $P = 3$. Scalar embedding + $\kappa_p$ is free in $P$ at no cost. |
| $G \cdot C[i,:]$ SC scaling, as in the paper | **rejected** | Assumes exactly one global coupling gain. |
| FiLM modulation of $\text{MLP}_{\text{SC}}$ by $m_{\text{agg}}$ | **removed** | Simplification. Consequence: template tokens become the primary structural route (§3.2). |
| $m_{\text{agg}}$ readout | **removed, then reinstated** | Dropped with the FiLM because it had no consumer, which was the wrong test. It carries coupling depth, the most discriminating structural feature in the survey, and now enters $g$ alongside $z_M$, $u$ and $o$. |
| Observation model as an out-of-band card field | **rejected** | The targets are statistics of BOLD, so the map from equations to statistics is not a function unless the network knows the observation model. It now enters through the `Out` node and two node-feature columns. |
| Observation flag on the observed `Deriv` root | **rejected** | Assumes the observable *is* a state variable. Jansen–Rit's is $y_1 - y_2$, an expression. A dedicated `Out` node attached to an arbitrary subexpression handles both. |
| CLS-only supervision for FC | **rejected** | Cannot support a correlation against arbitrary empirical FC. Bilinear head over region tokens instead. |
| Discarding $\widehat{\text{FCD}}$ after stage 1 | **rejected** | Route it through the shared FCD encoder, symmetrically with FC. Adds no information beyond $h_{\text{CLS}}$, but supplies the structure that makes the KS comparison direct — and stops stage 1 training a head that stage 2 ignores. |
| Freezing the backbone in stage 2 | **rejected** | Destroys the sim-to-real property that motivates the whole approach (§6.4). |
| Full statistic battery (~15 regional, lagged FC, edge-level FCD variance, FC spectrum, metastability, LZ complexity) | **cut** | Reduced to 6 / 1 / 103. Also removed the need for masked multi-task loss. |
| Raw constant-value column | **replaced** | Constants are now sampled log-uniformly over eight decades, so two `Const` nodes in one model can differ by $10^8$. Two columns instead: sign and $\log_{10}\lvert c\rvert$ (§2). |
| Three-column constant encoding `[sign, log(1+\|c\|), c/(1+\|c\|)]` | **rejected** | A generic recipe for a real of unknown distribution. The third column only repairs the resolution $\log(1+\lvert c\rvert)$ destroys below $\lvert c\rvert = 1$, which is half the prior. Knowing the prior is log-uniform removes the need. |
| State-variable identifier | **deferred** | Schema change with no forcing trigger yet. Structurally symmetric state-variable pairs are permitted into the corpus; if the encoder cannot separate them, that shows as a $V \ge 2$ transfer deficit. |
| Phase-1 weight transfer from the existing checkpoint | **abandoned** | The type vocabulary gained `Log` and `Out` regardless, so parity requires retraining either way. This is why the constant-encoding change costs nothing extra. |

---

## 9. Open questions

1. **Is a representation good for predicting simulated statistics the right initialization for predicting empirical discrepancy?** This is the load-bearing assumption and it is testable, not axiomatic. The experiment is the simulation-efficiency curve with the no-pretraining arm run explicitly.
2. **Do the four readouts in $g$ compose or interfere?** $g$ is now a sum of $\text{MLP}_g(\cdot)$, $z_M$, $u$, $o$ and $m_{\text{agg}}$. Each additional summand contributes less in expectation unless the MLPs rebalance, and $m_{\text{agg}}$ averages 1 to 2 nodes while $z_M$ averages 30 to 120. Cheap ablation: $g$ with and without $m_{\text{agg}}$, everything else fixed. Run it alongside the FiLM ablation.
3. **Is rank $k = 32$ enough** for the bilinear head to capture simulated FC? Sweep it; §6.5 is the diagnostic.
4. **Does removing FiLM cost accuracy at parity?** Phase 1 answers this directly — retrain MFM at $N=68$ and compare against current DELSSOME.
5. **Corpus balance over $P$.** Larger templates have more eligible coefficients, so the sampler may concentrate on $P = 4$–$5$ and leave $P = 2$ rare — which is where the mean-over-$p$ pooling is least exercised, since one of the two summands is always $\sigma$. Histogram $P$ over the first 50 models before generating any simulations. **Measured (2026-10-10)** over the valid corpus models: $P$ = 1/2/3/4/5 in 2/36/33/19/10%. The concentration feared here did not happen. $P = 2$ is the most common value, and $P = 1$ (fully noiseless models with one free coefficient) is rare.
6. **Flat draws in stage 1.** Draws flagged `flat` carry statistics computed from a signal with no variance, some of them NaN (§5.3). Decide whether stage 1 trains on them, excludes them like divergent draws, or labels them.
