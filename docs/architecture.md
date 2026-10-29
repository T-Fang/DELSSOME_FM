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

Indices: $i, j$ over regions; $p$ over regional parameters; $k$ over globals; $v$ over state variables; $m$ over DAG nodes.

---

## 2. Template encoder

A Graphormer over the equation DAG produces per-node embeddings.

$$H_{\text{tmpl}} = \text{Graphormer}(X_{\text{dag}}, A_{\text{dag}}) \;\in\; \mathbb{R}^{M \times d_t}$$

Four readouts are extracted by node role:

| Readout | Shape | Extracted from |
|---|---|---|
| $\kappa_p$ | $P \times d$ | the `Par` node for the $p$-th **regional** free parameter |
| $\lambda_k$ | $K \times d$ | the `Par` node for the $k$-th **global** free parameter |
| $u$ | $d$ | mean over the $V$ `Deriv` root nodes |
| $o$ | $d$ | the single `Out` node |
| $m_{\text{agg}}$ | $d$ | mean over the `Agg` nodes (1 or 2); zero if the model has none |
| $z_M$ | $d$ | mean over all $M$ nodes, a graph-level summary |

$\kappa_p$ is what makes the parameter encoder model-agnostic: it tags each parameter by *where in the equation it sits*, so the network learns "this is the coefficient multiplying a state inside a transfer function" rather than "this is column 2."

### Constant encoding

(Region-token scheme. The parameter-token scheme, §3.4, reads constants as values instead.)

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

$$s_k = W^{\text{glob}}\,\Psi[k] + \lambda_k, \qquad g = \text{MLP}_g\Big(\tfrac{1}{K}\textstyle\sum_k s_k\Big) + z_M + u + o + m_{\text{agg}} \;\in\; \mathbb{R}^{d}$$

$$\boxed{\;\text{token}_i = \big[\,t^{\text{par}}_i \,\|\, t^{\text{SC}}_i\,\big] + g \;\in\; \mathbb{R}^{d}\;}$$

Same mean-then-MLP construction on the global side, for the same reason.

### 3.4 Parameter tokens (`token_scheme: parameter`, decided by the project lead 2026-10-29)

An alternative to §3.1–3.3, chosen per run by `token_scheme`. The default, and every run before 2026-10-29, is `region`. The motivation: §3.1 averages a region's parameters before the transformer sees them, and §3.3 sums every global quantity into one offset. Both lose information. Two changes:

**Global values go into the template encoder.** Constants and parameters are no longer distinguished, only global and regional quantities.
- The Graphormer sees every `Const` node as a global `Par` node: type one-hot `Par`, scope −1, sign and $\log_{10}$ columns cleared.
- Every global node (each constant and each of the $K$ global parameters) adds $\text{MLP}_{\text{val}}(v)$ to its input embedding. $v$ is the value encoded as the sinh-policy parameters are, $\operatorname{asinh}(x/S)/Z$; $\text{MLP}_{\text{val}}$ is $1 \to d_t \to d_t$.
- The encoder therefore runs per draw, and $\kappa_p$ depends on the draw's global values.
- Regional `Par` nodes carry no value: theirs differ by region.
- Exact constant values are stored in the training store as `node_value.npy`. The clipped $\log_{10}$ column would lose the ~4% of sinh constants with $\lvert c\rvert < 10^{-4}$.
- The scheme requires a sinh-policy store; training refuses any other.

**One token per regional parameter per region,** with no pooling before attention and no global offset:

$$\text{token}_{i,p} = \text{MLP}_{\text{tok}}\big([\,W_\theta\,\Theta[i,p] \;\|\; \kappa_p \;\|\; C[i,:]\,]\big) \in \mathbb{R}^{d}, \qquad W_\theta \in \mathbb{R}^{d \times 1}$$

- One MLP ($2d + N \to d \to d$) is shared by all parameters and regions, giving $P \times N$ tokens: 136–340, since $P = 2..5$ in the sinh corpus.
- $\Psi$, $z_M$, $u$, $o$ and $m_{\text{agg}}$ no longer enter the tokens. The tokens get global information by attending to the template tokens (in-context conditioning).
- The transformer reads [CLS, $P \times N$ parameter tokens, $M$ template tokens], with a learned segment embedding each. Tokens of absent parameters are masked as attention keys.
- $r_i$ is the mean of region $i$'s $P$ outputs, absent parameters excluded. The heads are unchanged: regional statistics and the pair head read $r_i$; global statistics, FCD and the cost terms read $h_{\text{CLS}}$ (through the heads' predictions).

**Why $W_\theta$.** The project lead's sketch fed $\Theta[i,p]$ as one raw input among $1 + d + N$. $W_\theta$ gives the same functions, because the MLP's first layer is linear in $\Theta$ either way. But a raw scalar beside 196 other inputs starts with about a fifth of the weight. On one batch of draws of a single model (which differ only in their values), the raw form left the regional and FC-moment losses near their start after 300 steps. With $W_\theta$ it fit them as fast as region tokens did.

**To watch.**
- At initialisation the output is ~80× less sensitive to $\Psi$ than with region tokens, because $\Psi$ reaches the tokens only through attention. The coupling gains are global parameters and decide whether a draw is coupled, so whether training learns this dependence shows in the coupled-draw metrics.
- The sequence is $1 + 68P + M \approx 450$ tokens at $P = 5$, against ~180 for region tokens. A CPU training step is 1.5× slower.

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

Outputs used downstream: $h_{\text{CLS}} \in \mathbb{R}^{d}$ and $r_i \in \mathbb{R}^{d}$ for $i = 1..68$. Template output positions are discarded. (With parameter tokens, §3.4, the regions segment holds $P \times 68$ tokens, and $r_i$ pools region $i$'s.)

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

**Pairwise** — supervised on region-token pairs: $\operatorname{arctanh}(\text{FC}_{ij})$, over the 2278 upper-triangular edges. It is computed with a stable arctanh (`summary.stable_arctanh`), which clips |FC| ≥ 1 from float32 rounding to the largest value below 1, so the target is at most 8.66 in magnitude. About 5% of usable draws have such near-synchronised edges.

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

**No task masking is needed.** Every statistic here is universal — any model that produces BOLD produces all of them. The heterogeneous-label machinery from earlier drafts (firing rates for FIC but not Hopf, and so on) is unnecessary at this statistic set. The only exception is a divergent simulation, which has no statistics at all; those rows are excluded from stage-1 loss entirely rather than masked per-statistic. In the corpus they are flagged `diverged` and their statistics are NaN. They make up 14.9% of the draws of the train corpus models; flat draws (below) make up 1.7%.

Draws flagged `flat` are also stored: some region is quantised (SD ≤ 10⁻⁶ |mean|) or some statistic is non-finite (generation.md §6). Their statistics are meaningless where defined: a quantised region's FC is correlation between rounding noise. **They are excluded from the stage-1 loss like divergent draws (decided 2026-10-12, project lead).** A training row is a draw with `diverged` and `flat` both false.

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

### 7.1 Stage 1 as implemented (2026-10-14)

Code: `train/pack.py` (training store), `train/data.py`, `nn/` (network), `train/losses.py`, `train/stage1.py`; config `configs/stage1.yaml`. Choices this document leaves open, and the decisions taken while building it:

- **Training rows.** A draw of a corpus model that is neither diverged nor flat and whose FCD is complete (every window pair present). Draws with an incomplete or empty FCD are excluded (project lead, 2026-10-14). In the val split, 3.4% of usable draws had an incomplete FCD and 0.1% an empty one.
- **Target transforms, then z-scores** (project lead, 2026-10-14). The regional statistics become asinh(mean/SD), log10 SD, asinh(skew), log(excess kurtosis + 3), node strength and lag-1 autocorrelation. FC moments become the mean, log10 SD and skewness. The pairwise targets stay as the stable arctanh of FC.
  - Why transform: observables have arbitrary units, and raw regional SD spans 15 orders of magnitude. A plain z-score would put 60% of the values within 10⁻⁴ of each other, while 0.1% of rows supplied a third of the squared error.
  - z-score statistics come from the train split: per column for the regional statistics and the FC moments, and one value pooled over all edges for the pairwise targets.
  - The FCD CDF is *not* z-scored. The monotone head predicts it as a probability, and the uncertainty weighting sets its scale in the loss.
- **Inputs.** Θ and Ψ enter as log10 of the multipliers, so 10^U(−1, 1) becomes [−1, 1]. Absent parameters are 0 and masked. The structural connectivity C is the group SC divided by its maximum.
- **Template encoder (Graphormer).** Node input is the 21 features plus learned embeddings of in- and out-degree (clipped at 16). Attention gets a learned per-head bias for the undirected shortest-path length between nodes (clipped at 16, with one more bucket for "no path"). There are 4 pre-LN layers of width d_t = 128. The readouts are taken after projecting to d.
- **Sequence.** [CLS, 68 regions, M template tokens] with three segment embeddings. Padded template positions are masked out of attention.
- **Losses.** Mean squared error per group (regional, pairwise on 256 random edges per step, FCD, FC moments), combined by uncertainty weighting. Optimisation is AdamW with linear warm-up and cosine decay, bf16 mixed precision, and gradient clipping at 1.
- **Validation.** A fixed random subset of 50,000 val rows. Logged: MSE per group, their sum (`val/total`, which also selects checkpoints), and R² of the pairwise targets over all edges.

### 7.2 Stage-1 results so far (2026-10-16)

Four runs of 200,000 steps (about 2 h each on one H100 or H200) in MLflow experiment `stage1`; checkpoints in `outputs/stage1/<run>/`. "Best" is the lowest val/total.

| Run | FC offset | Dropout / weight decay | Best val total | Val pairwise | R² pairwise |
|---|---|---|---|---|---|
| `stage1_v1` | no | 0 / 0.01 | 1.251 | 0.422 | 0.585 |
| `stage1_reg` | no | 0.1 / 0.05 | 1.232 | 0.426 | 0.582 |
| `stage1_offset` | yes | 0 / 0.01 | 1.169 | 0.351 | 0.656 |
| `stage1_offset_reg` | yes | 0.1 / 0.05 | **1.162** | 0.361 | 0.646 |

Against `stage1_v1`, with every model weighted equally and a 95% CI from resampling models: `stage1_offset_reg` is better on val by 0.075 [0.019, 0.128] and on unseen models by 0.21 [0.02, 0.43]. `stage1_offset` is better on val by 0.070 [0.026, 0.116]. Regularisation alone is not significant.

**Decision (project lead, 2026-10-16).** Keep the original implementation as the default (`configs/stage1.yaml`: no FC offset, dropout 0, weight decay 0.01): the gains are small, so simplicity wins. The offset remains available as `fc_offset`. Next: a scaling sweep over training models and draws per model (`train_models`, `draws_per_model`), then the corpus composition (coupled models).

**Where the error is** (generalisation diagnostic, `sim/generalisation.py`; 500 train and 500 val models, new draws 100–109, each with a seen and an unseen SC):

- **Parameters and connectomes generalise.** New draws of training models are predicted as well as training rows: total 0.51 against 0.53 for `stage1_v1`. An unseen SC adds only about 0.01.
- **Equations do not generalise.** Unseen models cost 2.6× as much: total 1.35, pairwise R² 0.57 against 0.78. The train/val gap is generalisation across equations.
- **Error is concentrated in a few models.** The hardest 5% of val models carry 56% of the error (per-model total: median 0.22, 95th percentile 4.9). They are the coupled models: median per-model error is 0.12 at mean FC < 0.05, 0.37 at 0.05–0.5, 1.77 at 0.5–0.9, and 1.34 at 0.9 or above. V = 2 makes up 66% of them, and their DAGs are larger (median 49 nodes against 36). Their error is pairwise (42%) and FC moments (35%).
- **Report figure** (`generalisation.png`; `scripts/generalisation_report_data.py`, then `plot_stage1_report.py --generalisation`). It shows a 2 × 2 matrix of equations (seen / new) × SC (seen / new). Each cell is the mean of per-model totals, so every model counts equally; the bullets above weight rows. The CI is a 95% bootstrap over models. The seen-equation cells keep only models with training rows in that checkpoint's training set; for a 10-draw run this drops 6–7 of the 477 models. `stage1_v1`: 0.63, 0.67 (seen equations; seen / new SC); 1.55, 1.57 (new equations). `d10_n300k_mlp_cat`: 0.60, 0.62 and 1.32, 1.34. Three times as many models cut the cost of new equations by about 0.23 and left seen equations almost unchanged. The gap is still about 2.2×.

### 7.3 Scaling and composition (2026-10-17 to 2026-10-19)

**Scaling.** All runs use the stage-1 defaults and 200,000 steps, and report the best val total. With all draws (~80 per model): 10k models 1.867, 30k 1.529, 100k 1.251 (`stage1_v1`). With 10 draws per model: 10k 1.927, 30k 1.644, 100k 1.282, then on the 10-draw pool (`configs/corpus_d10.yaml`, `training_store_d10`) 150k 1.163 and **300k 1.068** (pairwise R² 0.636). Distinct models matter much more than draws per model, and the curve has not flattened at 300k. The project lead therefore decided on 10 draws per model (2026-10-17).

**Composition** (150k models, 10 draws; "coupled" means the model's mean FC over its draws is above 0.05; the pool is 36.4% coupled). Paired per-model differences from the natural share on val (20k rows), with 95% CIs over models (`scripts/compare_by_regime.py`):

| Coupled share | Uncoupled models (5,003) | Coupled models (2,943) |
|---|---|---|
| 36.4% (natural; reference) | 0.799 | 1.995 |
| 50% | +0.053 [+0.004, +0.100] | −0.075 [−0.202, +0.048] |
| 70% | +0.138 [+0.080, +0.195] | −0.118 [−0.236, −0.002] |
| 85% | +0.244 [+0.175, +0.315] | −0.134 [−0.269, +0.000] |
| 300k models, natural | −0.060 [−0.101, −0.023] | −0.159 [−0.270, −0.049] |

Oversampling coupled models buys a small, saturating gain on coupled models (at most about 7%) at a steep cost on uncoupled ones (up to 31%). Doubling the number of models at the natural composition improves both regimes, and helps coupled models more than any oversampling does.

### 7.4 Zero-shot test on the reference models (2026-10-19)

MFM, FIC and Hopf at their 50 published test fits (`sim/reference_eval.py`, `configs/reference_eval.yaml`):

- **Setup.** Each card is re-expressed in the corpus convention (constant = signed median of the fits, multiplier around 1; equations unchanged, tested). MFM keeps 36 sets: 14 have a drive whose sign changes across regions. Each set is simulated twice with the corpus protocol; under it the sets keep their original-protocol mean FC (MFM 0.23, FIC 0.49, Hopf 0.30).
- **Results for `d10_n300k_natural`** (medians over sets; the noise floor is a second simulation, the baseline is the training average):

| Model | 1 − r: model / floor / baseline | d: model / floor / baseline | KS: model / floor / baseline |
|---|---|---|---|
| MFM | 0.32 / 0.08 / 0.54 | 0.17 / 0.01 / 0.12 | 0.34 / 0.09 / 0.72 |
| FIC | 0.99 / 0.06 / 0.49 | 0.13 / 0.03 / 0.36 | 0.89 / 0.14 / 0.74 |
| Hopf | 0.18 / 0.05 / 0.44 | 0.09 / 0.03 / 0.17 | 0.53 / 0.16 / 0.74 |

- **The predicted FC is nearly constant** (≈ 0.39, i.e. z ≈ 0): its range within a set is 0.03 for MFM, 0.003 for Hopf and 0.0004 for FIC. Its pattern still correlates with the simulation for MFM and Hopf. With the FC offset (100k-model runs), FIC's 1 − r drops to 0.44, but the other terms are mixed. Every variant stays far above the noise floor, which is what stage 2 is for.

### 7.5 FC heads and the cost loss (2026-10-19)

Same setup (300k models × 10 draws, 200k steps). Each run is evaluated on the same 200k test rows (`final/test/*`, the final evaluation of `train/fit.py`); the cost terms are medians between prediction and simulation, in raw units (`scripts/comparison_table.py`).

#### Held-out test models (synthetic; medians of cost terms over draws)

| model | MSE total | R² FC | 1−r coupled | d coupled | KS coupled | d uncoupled | KS uncoupled |
|---|---|---|---|---|---|---|---|
| 100k models × ~80 draws, bilinear head | 1.266 | 0.575 | 0.728 | 0.199 | 0.287 | 0.386 | 0.046 |
| 300k models × 10 draws, bilinear head | 1.083 | 0.631 | 0.724 | 0.190 | 0.286 | 0.386 | 0.050 |
| 300k × 10, bilinear head + cost loss | 1.076 | 0.628 | 0.549 | 0.183 | 0.261 | 0.386 | 0.040 |
| 300k × 10, pair-MLP head (symmetric) | 1.033 | 0.679 | 0.555 | 0.053 | 0.287 | 0.004 | 0.047 |
| 300k × 10, pair-MLP head (concat) | 1.009 | 0.690 | 0.590 | 0.045 | 0.271 | 0.005 | 0.044 |

#### Reference models at their published test fits (medians over sets)

| model | MFM 1−r | MFM d | MFM KS | FIC 1−r | FIC d | FIC KS | Hopf 1−r | Hopf d | Hopf KS |
|---|---|---|---|---|---|---|---|---|---|
| 100k models × ~80 draws, bilinear head | 0.423 | 0.161 | 0.894 | 1.042 | 0.131 | 0.897 | 0.291 | 0.084 | 0.631 |
| 300k models × 10 draws, bilinear head | 0.323 | 0.168 | 0.337 | 0.992 | 0.130 | 0.888 | 0.181 | 0.085 | 0.530 |
| 300k × 10, bilinear head + cost loss | 0.401 | 0.164 | 0.355 | 0.522 | 0.130 | 0.874 | 0.238 | 0.087 | 0.318 |
| 300k × 10, pair-MLP head (symmetric) | 0.500 | 0.097 | 0.521 | 1.159 | 0.495 | 0.899 | 0.378 | 0.252 | 0.750 |
| 300k × 10, pair-MLP head (concat) | 0.365 | 0.147 | 0.563 | 0.453 | 0.485 | 0.901 | 0.192 | 0.261 | 0.705 |
| a 2nd simulation of the same parameters | 0.077 | 0.011 | 0.092 | 0.058 | 0.029 | 0.139 | 0.048 | 0.030 | 0.156 |

**Synthetic test models.** The pair-MLP heads beat the bilinear head on every term. `mlp_concat` is best overall, and both remove the FC-level bias (*d* on uncoupled draws from 0.39 to 0.005). The cost loss improves 1 − r and KS but keeps the bilinear head's *d* bias. **Reference models.** None of the gains carries over consistently. The MLP heads predict FIC's FC level near 0 (*d* 0.49), and the cost loss is best on Hopf KS and FIC 1 − r. All runs remain far above the noise floor: zero-shot transfer to the real models is the open problem.

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
6. ~~**Flat draws in stage 1.**~~ Decided 2026-10-12: excluded from the stage-1 loss, like divergent draws (§5.3).

### 7.6 The sinh corpus: scaling, targets, heads, size and coupled-only training (2026-10-25 to 2026-10-27)

The corpus of generation.md §4 (`sinh` policy; `corpus_sinh`, `training_store_sinh`: 100,000 train models, 4.04 M usable rows, 40% of draws; val and test 10,000 models, 0.41 M rows each). Every run: 200,000 steps, batch 256, the concatenation pair-MLP FC head (the "original" model from here on), parameters encoded as asinh(x / 1e-4) / 19.1. Metrics are each run's final evaluation (`final/<split>/*`, 200,000 rows per split; train = the rows it trained on). Tables and figures: `outputs/report_sinh/` (`comparison_table.md`, `figures/scaling_grid.png`, one folder per run). Decided along the way: the original model uses the MLP FC head; GPU jobs go to L40S; new variants train with the cost loss (2026-10-26).

**The new corpus is much harder to fit.** The original model reaches val total 1.79 and trained-rows total 1.48 (old corpus, same head: 1.0 and 0.49). Trained rows and test differ little in the cost terms (coupled 1 − r 0.62 against 0.64, KS 0.41 against 0.43): the network does not fit the dynamics even of the models it trained on. Every draw now has its own parameter scale, and regions within a draw differ.

**Scaling** (original + cost loss, `sinh_m<M>k_d<D>_cost`). Val total: 10k models 2.36 / 2.27 / 2.22 at 10 / 30 / 100 draws; 30k 2.18 / 2.06 / 2.01; 100k 1.96 / 1.87 / 1.83. More models helps most; draws help too, less. Trained-rows totals stay at 1.5–1.9 at every scale.

**Model size** (all + cost loss, 100k models × all draws):

| model | parameters | trained rows | val | test | test R² FC | test coupled 1 − r |
|---|---|---|---|---|---|---|
| d 128, 4 + 4 layers (`sinh_m100k_d100_cost`) | 1.8 M | 1.55 | 1.83 | 1.87 | 0.46 | 0.59 |
| wide: d 256, 4 + 4 | ~6.8 M | 1.53 | 1.80 | 1.88 | 0.45 | 0.60 |
| deep: d 128, 8 + 8 | ~3.4 M | 1.54 | 1.81 | 1.87 | 0.46 | 0.59 |
| wide + deep: d 256, 8 + 8 | ~13 M | 1.49 | 1.80 | 1.84 | 0.47 | 0.59 |

Seven times the parameters changes the fit to the training rows by 0.07. Capacity is not what limits the fit.

**Targets and heads** (test; total = the four statistic groups):

- Cost loss: total 1.83 → 1.87 and R² 0.49 → 0.46, but coupled 1 − r 0.64 → 0.59 and uncoupled d 0.016 → 0.001.
- catch24 (`catch24_loss`): adding it leaves the statistics as they were (original + catch24: 1.82; + cost + catch24: 1.87). catch24 MSE 0.61 (z-scored; trained rows 0.55); per-feature R² 0.0–0.66, median 0.48. The time-reversal statistic (`CO_trev_1_num`) has R² ~ 0: in these noise-driven simulations it is sampling noise. catch24 alone trains to the same catch24 MSE (0.60).
- Latent FC head (FC autoencoder, latent 128; the frozen decoder reconstructs val FC to R² 0.968 and coupled 1 − r 0.12): worse than the pair MLP (coupled 1 − r 0.79 against 0.59). Predicting the latent vector from h_CLS is the hard part, not decoding.

**Network FC** (Yeo-7, data/networks.py: the FC of each network's averaged raw signals, 21 values; 50,000 rows per split, identical across arms):

| arm | test coupled 1 − r | test MSE (arctanh) | test d uncoupled | trained-rows coupled 1 − r |
|---|---|---|---|---|
| original (`sinh_orig`), region predictions downsampled | 0.48 | 1.73 | 0.12 | 0.47 |
| network FC only (`sinh_fcnet_only_v2`), h_CLS head | 0.37 | 1.74 | 0.05 | 0.35 |
| original + network FC (`sinh_orig_fcnet`), its head | 0.36 | 1.67 | 0.04 | 0.34 |
| original + network FC, downsampled | 0.47 | 1.77 | 0.12 | 0.46 |

Predicting network FC directly gives a better network pattern than downsampling region predictions (1 − r 0.36 against 0.47), and adding it does not hurt the region-level statistics (test total 1.86). But the MSE is no better, and training rows fit no better than test: the lower resolution makes the target easier to state, not the dynamics easier to learn. A small region-level FC bias (d 0.016) adds up to 0.12 in network FC, because a network signal sums many region pairs.

**Coupled draws only** (`sinh_coupled_only`, + cost loss; trained and validated on draws with mean FC > 0.05: 0.83 M rows from 68,686 models). Against the same-size model trained on all draws, on the same coupled rows (`final_coupled/*`):

| model | rows | total | R² FC | 1 − r | d | KS |
|---|---|---|---|---|---|---|
| all draws (`sinh_m100k_d100_cost`) | test, coupled | 5.70 | 0.24 | 0.59 | 0.16 | 0.44 |
| all draws, wide + deep | test, coupled | 5.61 | 0.25 | 0.59 | 0.15 | 0.43 |
| coupled draws only | test, coupled | **4.56** | **0.42** | **0.56** | **0.09** | **0.40** |
| all draws | trained, coupled | 4.76 | 0.38 | 0.58 | 0.12 | 0.42 |
| coupled draws only | trained, coupled | 3.44 | 0.58 | 0.54 | 0.07 | 0.37 |

Training only on coupled draws improves every term on coupled draws, by more than any change of size, head or target: the uncoupled majority (79% of rows) interferes with learning coupled FC. It also transfers to the reference models, which are all coupled: KS 0.66 / 0.58 / 0.38 for MFM / FIC / Hopf against 0.89 / 0.89 / 0.70, and d 0.11 / 0.31 / 0.04 against 0.25 / 0.49 / 0.28. Its checkpoint is from step 55,000 (0.83 M rows: it overfits sooner). It cannot be used alone, since whether a draw is coupled is known only after simulating it: hence the proposal to predict coupling from the parameters and condition the network on it.
