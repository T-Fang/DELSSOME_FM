# DELSSOME-FM: proof-of-concept synthetic model generation

**Scope.** Five reference models: Linear/OU, Wilson–Cowan, MFM, FIC, Hopf. The template additionally reaches Montbrió–Pazó–Roxin, FitzHugh–Nagumo and Hindmarsh–Rose, and with filter pairs (currently out of the sampler) Jansen–Rit and Wendling.

**Status.** Replaces §4 of the v2 design doc. That document stays as the reference for later stages; none of it is withdrawn.

**Implementation (2026-10-10).** The template, sampler, simulator and corpus builder are implemented (`src/delssome_fm/spec/`, `src/delssome_fm/sim/`). Seven reference cards are hand-written: the five here plus MPR and Jansen–Rit, which were added on 2026-10-05. Decisions taken while building are dated in the sections they change. "As implemented" paragraphs record details this document left open. Generation of the corpus started on 2026-10-10 (§9).

---

## 1. The template

For each state variable $v \in \{1 \dots V\}$ in region $i$:

$$\frac{dx^v_i}{dt} \;=\; \underbrace{-\sum_u L_{vu}\,x^u_i \;+\; \beta_v}_{\text{affine}} \;+\; \underbrace{\gamma_v\,g_v(\mathbf{x}_i)\,\Phi_v\!\left(u^v_i\right)}_{\text{nonlinear term, optional}} \;+\; \underbrace{\sum_{c\,\in\,D_v} A_{c,i}}_{\text{coupling}} \;+\; \underbrace{\sigma_v\,\nu^v_i(t)}_{\text{noise, may be zero}}$$

$$u^v_i \;=\; \sum_u w_{vu}\,x^u_i \;+\; I_v \;+\; \sum_{c\,\in\,U_v} A_{c,i}$$

$$A_{c,i} \;=\; G_c\sum_j C_{ij}\Big(f_c\big(p^c_j\big) \;-\; \delta_c\,f_c\big(p^c_i\big)\Big), \qquad p^c_i \;=\; \sum_u m^c_u\,x^u_i$$

$$o_i \;=\; x^{q}_i \quad\text{or}\quad x^{q}_i - x^{q'}_i \qquad\text{(the observable)}$$

### Reading it

Four questions per state variable:

1. What does the region do on its own, affinely? → $L$, $\beta_v$
2. Does it have a nonlinear term, and what is it? → $\gamma_v$, $g_v$, $\Phi_v$
3. Which coupling channels feed it, and where? → $D_v$, $U_v$
4. How much noise? → $\sigma_v$

Plus two model-level questions: what the channels transmit, and what is observed.

**Coupling channels are declared once and referenced.** A channel is a named quantity any slot may reference. Referenced from $u^v$ it lands under $\Phi$ (deep in the equation graph: MFM, FIC, Wilson–Cowan). Referenced from $D_v$ it lands at the derivative (shallow: Linear, Hopf). Same object, two positions, no switch in the equation. Injection is **set membership**, not a coefficient, so the injection site lives in the graph's topology rather than hidden in a number that might be near zero.

**The gate is the self-limiting mechanism**, and its options are the ways these models keep themselves bounded: receptor saturation $(1-S)$ in Wong–Wang, refractoriness in Wilson–Cowan, amplitude saturation $\lvert z\rvert^2$ in Hopf, spike-versus-reset balance in MPR.

**Three places say "a weighted combination of state variables, optionally through a nonlinearity":** the transfer input $u^v$, the channel source $p^c$, and the observable $o$. The same object applied to yourself, to a neighbour, and to the observer.

### The five reference models

| | $L$ | channel | $\Phi_v$ | gate $g_v$ | injection | observable |
|---|---|---|---|---|---|---|
| **Linear** | $1/\tau$ | $x$, direct | no term | — | $D_1$ | $x$, BW |
| **MFM** | $1/\tau_S$ | $S$, direct | $H$ | $1-S$ | $U_1$ | $S$, BW |
| **FIC** | $\mathrm{diag}(1/\tau_E, 1/\tau_I)$ | $S^{(E)}$, direct | $H$, $H$ | $1-S^{(E)}$, none | $U_E$ | $S^{(E)}$, BW |
| **W–C** | $\mathrm{diag}(1/\tau_e, 1/\tau_i)$ | $E$, direct | logistic ×2 | $1-E$, $1-I$ | $U_E$ | $E$, BW |
| **Hopf** | $\begin{pmatrix}-a & \omega\\ -\omega & -a\end{pmatrix}$ | $x$ and $y$, diffusive | identity | $x^2+y^2$ | $D_x$, $D_y$ | $x$, direct |

Also reachable and not designed for: **MPR** ($\beta_v$, bare-state gate, free $s_u$, $\sigma = 0$), **FitzHugh–Nagumo** and **Hindmarsh–Rose** (one-hot quadratic gate).

---

## 2. What the generator samples

| Slot | Options | Sampling |
|---|---|---|
| $V$ | 1 to 6 | 0.30 / 0.45 / 0.15 / 0.10 over $\{1\}$, $\{2\}$, $\{3\}$, $\{4,5,6\}$ |
| $L$ diagonal | leak (positive) or growth (negative) | always present |
| $L$ off-diagonal | any real | present with probability ~0.5, then masked |
| $\beta_v$ | present or absent | |
| nonlinear term | present or absent | present ~0.85 |
| $\gamma_v$ | gain | always a fixed constant |
| $g_v$ | $1$; $1-x^v$; $x^u$ (any $u$); $\sum_u s_u(x^u)^2$ | |
| $s_u$ | any real | 40% isotropic (all 1), 30% one-hot, 30% free signed |
| $\Phi_v$ | identity, logistic, Wong–Wang $H$ | canonical, no constants (§3) |
| $w_{vu}$ | any real | present with probability ~0.5, then masked |
| $I_v$ | present or absent | |
| channels | 1 or 2 | 1 at ~0.85, 2 at ~0.15 |
| $m^c_u$ | one-hot | |
| $f_c$ | identity, logistic, Wong–Wang $H$ | identity ~0.85 |
| $\delta_c$ | 0 direct, 1 diffusive | |
| $G_c$ | free global, one per channel | |
| injection | $U_v$ (current) or $D_v$ (derivative) | |
| $\sigma_v$ | free, or zero | zero ~0.1 |
| observable | single variable, or difference of two | single ~0.9 |
| observation | Balloon–Windkessel, direct downsample | independent of everything else |

**As implemented** (`spec/sampler.py`, where each such constant is marked "choice"). The table leaves four probabilities open, and each is fixed at 0.5: $\beta_v$ present, $I_v$ present, $\delta_c$ diffusive, and injection into $U_v$ rather than $D_v$ when $u^v$ exists. The two "present ~0.5" masks of $L$ and $w$ are also 0.5.

### Why $L$ and $w$ are dense with random masking

An earlier version sampled from a menu of sign patterns (push–pull, both excitatory, cascade, selector). Those were four motifs noticed by hand, which is exactly the prior a model-agnostic encoder should not carry. Sampling every entry and then masking recovers all four as particular masks, without making them the only reachable structures.

### Two constraints on the sampler

**Two channels must differ** in at least one of source variable, $\delta_c$, or injection site. Two identical channels are one channel with double the gain, and the encoder would see a redundant subgraph.

**A difference observable needs structurally distinguishable variables.** If the two share a transfer function, a gate, and a row of $L$, their difference is near zero and the observable is noise. Cheap to check at sampling time, and it prevents a failure the screen would only catch after a full simulation.

### How each slot extends

| Slot | Extends to | Unlocks |
|---|---|---|
| $L$ off-diagonal | filter pairs (critically damped $\{-1;\ a^2,\ 2a\}$) | Jansen–Rit, Wendling, LaNMM |
| $\Phi_v$, $f_c$ | polynomial, softplus | Zerlaut–Destexhe |
| $g_v$ | conductance driving force | Larter–Breakspear, Liley |
| nonlinear term | several per variable | Larter–Breakspear, Liley |
| $\delta_c$ | normalised, delayed | Robinson, delayed W–C |
| injection | into a gate | Larter–Breakspear |
| observable | weighted combination, nonlinearity, several outputs | EEG lead field, multimodal |
| observation | expanded into the DAG as extra state variables | regional hemodynamics, EEG |

Every excluded model enters by lengthening a list, not by restructuring. **Kuramoto is the one genuine wall**: its state lives on a circle rather than in $\mathbb{R}^V$, which is a different state space, not a slot value.

---

## 3. Transfer functions are canonical

Published models use visibly different sigmoids (Wilson–Cowan's $a_k, \theta_k$; Jansen–Rit's $2e_0, r, v_0$; Wong–Wang's $a, b, d$). **None of those constants is a free direction.**

$$\gamma_v\;g_v\;\underbrace{M\,\sigma\big(\rho(u^v-\theta)\big)}_{\Phi_v}, \qquad u^v = \sum_u w_{vu}x^u + I_v + \sum_{c\in U_v}A_c$$

| Parameter | Degenerate with | Absorbed into |
|---|---|---|
| output scale $M$ | the gain $\gamma_v$ | $\gamma_v \leftarrow M\gamma_v$ |
| slope $\rho$ | everything inside $u^v$ | $w_{vu}, I_v, G_c \leftarrow \rho\,(\cdot)$ |
| threshold $\theta$ | the drive $I_v$ | $I_v \leftarrow \rho(I_v - \theta)$ |

So the library is three entries with no constants at all:

| $\Phi$ | Form |
|---|---|
| identity | $u$ |
| logistic | $1/(1+e^{-u})$ |
| Wong–Wang | $u/(1-e^{-u})$ |

**Also absorbed, and therefore not separate entries.** $\tanh u = 2\sigma(2u) - 1$: the slope 2 goes into $w$ and $I$, the scale into $\gamma_v$, the $-1$ into $\beta_v$ and $L$ (valid when $g_v$ is affine in the states, which covers $1$, $1-x^v$ and $x^u$). Wilson–Cowan's zero-shifted sigmoid $\sigma(a(z-\theta)) - \sigma(-a\theta)$ absorbs the same way. The proof of concept uses the **non-shifted** logistic for Wilson–Cowan, which means a region with no input has a small positive baseline rather than exactly zero: a slightly different resting fixed point, nothing structural.

**Three consequences.** Two degeneracy rules disappear (nothing left to fix). The equation graph canonicalises, so two models differing only in a sigmoid's slope now compile to the same subgraph. And MFM and FIC, which appear to have three different transfer functions between them, turn out to share one.

---

## 4. Constants and free parameters

### Constants

**One flat prior, no tiers, no per-slot ranges:**

$$\log_{10}\lvert c\rvert \sim \mathcal{U}[-4,\, 4], \qquad \operatorname{sign}(c) \text{ from the slot's allowed signs}$$

Eight decades, sampled log-uniformly. An earlier version had hand-drawn per-slot ranges split into a neural tier and a BOLD tier. Both are gone, for the same reason the sign-pattern menu went: ranges inferred from five examples are a prior about which models exist, and a model-agnostic encoder should not carry one.

**Two things this costs, both accepted.**

Most sampled models will have one term dominating the others. Constants are drawn independently, so with $n$ of them the chance they all land within two decades of each other is roughly $(2/8)^{n-1}$: about 6% at $n=3$ and under 1% at $n=5$. The dominated models are not divergent and not flat, so they pass the screen and enter the corpus as expensive near-duplicates of simpler models. The response is to budget for a low yield rather than to add screening.

Raw constant magnitudes now span $10^8$ within a single model. That is why the encoder's constant column is $\log_{10}\lvert c\rvert$ rather than the raw value (architecture doc §2).

**One failure the screen will not catch.** A model whose dominant rate falls below about $10^{-3}$ s⁻¹ does not equilibrate within a 15-minute scan. Its BOLD is a slow drift, which has nonzero variance, so the flatness check passes, but every region drifts together and mean off-diagonal FC comes out near 1. No screen for it, but **log mean off-diagonal FC per model** when building the training data. It is free, and a value near 1 across all parameter draws identifies these immediately.

### Which coefficients can be free

**Step 1, strike the degenerate.**

| Pair | Rule |
|---|---|
| $\Phi$'s scale, slope, threshold | nothing to strike; the library is canonical |
| $f_c$'s slope and $\lVert m^c\rVert$ vs $G_c$ | fix the slope and $\lVert m^c\rVert$ |
| quadratic gate's overall scale vs $\gamma_v$ | fix the first nonzero $s_u = 1$ |
| $\gamma_v$ | never a candidate |
| $\Phi = $ identity **and** $g_v = 1$ | forbid; the term is affine and absorbed by $L$, $\beta_v$, $D_v$. Omit it instead |

**Step 2, two forced assignments.** $\sigma_v$ is always regional. Every coupling gain is always global.

**Step 3, sample from what survives.**

$$\text{candidates} \;=\; \{L_{vu}\} \;\cup\; \{\beta_v\} \;\cup\; \{w_{vu}\} \;\cup\; \{I_v\}$$

Take $n \in \{1,2,3,4\}$, giving $P = 1 + n$ and $K$ = number of channels. An unselected candidate becomes **one scalar shared by all 68 regions**, like $w_{II} = 1$ in FIC. So the split is binary in the DAG: selected candidates compile to `Par` nodes, everything else to `Const` nodes, and $\Theta$ has exactly $P$ columns.

As implemented, $n$ is capped at the number of candidates, and $\sigma$ is absent when every variable is noiseless. So $P$ ranges from 1 to 5. Among valid corpus models (§9), $P$ = 1/2/3/4/5 occurs in 2/36/33/19/10% and $K$ = 1/2 in 86/14%.

$P$ and $K$ vary by model deliberately. All five reference models are $P{=}3$, $K{=}1$, so a uniformly $P{=}3$ corpus would never exercise the pooling that makes the encoder parameter-count-agnostic, and that failure would stay invisible at test time.

Regional values are drawn i.i.d. per region. The known cost of dropping spatial structure is that simulated FC leans more on SC than it would with realistic parameter maps.

### Values of the free parameters (decided 2026-10-05)

A selected candidate still gets a sampled nominal constant c, drawn from the constant prior above (log-uniform over 8 decades, sign from the slot). The card stores the slot as `scale × parameter` with scale = c. $\Theta$ and $\Psi$ then hold **dimensionless multipliers**:

$$\Theta[i,p] \sim 10^{\,\mathcal{U}[-1,\,1]} \text{ i.i.d. over regions and draws}, \qquad \Psi[\kappa] \sim 10^{\,\mathcal{U}[-1,\,1]}$$

so the effective coefficient is $c\,\Theta[i,p]$, within one decade of its nominal value. The magnitude reaches the encoder through the `Const` node's $\log_{10}|c|$ feature, and $\Theta$ stays $O(1)$ for the network.

**Coupling gains are relative, not flat (decided 2026-10-05).** The nominal gain of channel $c$ is drawn relative to the term it competes with at its injection site:

$$G_c = 10^{\,\mathcal{U}[-2,\,1]}\;\frac{\text{reference}}{\overline{\textstyle\sum_j C_{ij}}}, \qquad \text{reference} = \begin{cases}|L_{vv}| & \text{injected at } dx^v/dt\ (D)\\ \max(|w_{vu}|, |I_v|) & \text{injected into } u^v\ (U)\end{cases}$$

where $\overline{\sum_j C_{ij}} = 0.36$ is the mean row sum of the training group SCs rescaled to max 0.02. Coupling therefore ranges from 1% to 10× the competing term. With the flat 8-decade prior, coupling was almost always negligible: in the first 60-candidate pilot, 6 of the 7 usable kept models had mean off-diagonal FC ≈ 0, so their FC carried no connectome structure and the stage-1 pairwise targets were noise. With the relative prior, a 300-candidate screen kept 19.7% (was ~13-15%). Of the kept models, 63% still had |mean FC| < 0.05, 24% were between 0.05 and 0.9, and 14% were above 0.9. The coupling target could reach the observable in 57 of 59 kept models, so the remaining FC ≈ 0 is weak coupling, not structure: mean FC becomes substantial only near the critical coupling ratio of about 1. The project lead kept the ratio range $10^{\mathcal{U}[-2,1]}$ (2026-10-05). At corpus scale (397,500 candidates screened by 2026-10-10), the valid models' mean off-diagonal FC is below 0.05 in 61%, between 0.05 and 0.9 in 32%, and above 0.9 (the slow-drift signature above) in 6%.

**One shared $\sigma$.** $\sigma$ is a single regional parameter (the "1" in $P = 1 + n$). Every noisy variable uses it through its own sampled scale constant, $\sigma_v = c_v\,\sigma$. A variable is noiseless with probability ~0.1.

### Every reference model lands on the same triple

| | regional | global |
|---|---|---|
| MFM | $w$, $I$, $\sigma$ | $G$ |
| FIC | $w_{EE}$, $w_{EI}$, $\sigma$ | $G$ |
| Wilson–Cowan | $c_1$, $P$, $\sigma$ | $c_5$ |
| Hopf | $a$, $\omega$, $\sigma$ | $g$ |
| MPR | $\bar\eta$, $J$, $\sigma$ | $G$ |

Recurrent gain, drive, noise. Five times, across four traditions. Two of the five are our choice rather than a published fit, but the natural choice landing in the same place each time was not engineered.

---

## 5. Sampling rules

Three rules, all cheap and all applied before any simulation.

**R1. Dissipativity, on the eigenvalues.** With dense off-diagonals, stability is a property of $L$'s spectrum, not its diagonal: an all-positive diagonal can still leave $-L$ expansive. Require $\min_k \mathrm{Re}\,\lambda_k(L) > 0$ unless at least one variable carries a quadratic gate to supply saturation. A $6\times6$ eigendecomposition costs microseconds and is strictly better than the diagonal check.

Without this, the acceptance rate drops noticeably once off-diagonals are dense, and every rejection comes from the expensive simulation screen rather than the free symbolic one.

**R2. No affine-only nonlinear term.** If $\Phi_v$ is the identity and $g_v = 1$, the term is affine and fully absorbed by $L$, $\beta_v$ and $D_v$. Omit the term instead. This is also the correct card for the Linear model, whose earlier listing (identity transfer, no gate) was the forbidden combination and worked only because its channel injects at the derivative.

**R3. Quadratic gate normalisation.** Fix the first nonzero $s_u = 1$, since the overall scale is degenerate with $\gamma_v$.

---

## 6. The one screen

Simulate 68 regions with the model's connectome (§9) and four random parameter draws, roughly three minutes of simulated time each.

- **Reject** if every draw diverges (NaN, or state past a fixed bound).
- **Reject** if every draw's **observed signal** is flat (temporal variance below a floor).
- **Keep** otherwise.

No parameter-box calibration, no cost-landscape screening, no stratification, no fingerprint deduplication. Four draws surviving is enough.

**As implemented** (`sim/corpus.py::screen`, `configs/corpus.yaml`):

- Each screen draw runs 250 frames at TR 0.72 s (180 s) after 50 frames of burn-in, from $x_0 \sim \mathcal{U}(-0.1, 0.1)$ per state.
- **Diverged** means any state or recorded value is non-finite or beyond ±10⁶ at a frame boundary (`sim/integrate.py`). The flag is sticky.
- **Flat** means every region's temporal SD of the recorded signal is at most $10^{-5}\max(1, |\text{mean}|)$.

At corpus scale, 16.2% of candidates pass the screen. Of the rejections, 99% are "every draw diverged".

**Screen the observable, not the state.** A model can have perfectly healthy dynamics and a dead observable: a difference of two near-identical variables cancels, or a single-variable output selects something with no variance. That failure is invisible to a state-level check and would put flat BOLD and meaningless FC into the corpus.

---

## 7. The generation flow

```
BUILD THE CORPUS
  repeat until we have ~100,000 models:

      pick the observation model
      pick the number of state variables

      sample L densely, then mask off-diagonals
      check the eigenvalue condition        (R1) -> resample if it fails

      for each variable:
          pick whether it has a nonlinear term, and if so:
              a transfer function, a gate, input weights, a drive
          pick whether it carries noise

      declare 1 or 2 coupling channels:
          what each transmits, direct or diffusive, its gain
          which slots each one is injected into

      pick the observable: one state variable, or a difference of two
      pick the model's connectome: one of its split's group SCs (64 for train)
      fill in the constants: log-uniform over 8 decades, signs per slot
      work out which coefficients are eligible to be free, pick a few

      simulate 68 regions, 4 random parameter draws
      apply the observation model
      if every draw blew up, or every observed signal was flat  ->  discard
      otherwise                                                 ->  keep

BUILD THE TRAINING DATA
  for each model we kept:
      repeat ~100 times:
          draw regional parameters and coupling gains
          simulate -> observation model -> BOLD at TR
          compute the summary statistics
          store (model, parameters, connectome, statistics)
      divergent runs are stored too, labelled as divergent
```

Essentially all the compute is in the second loop. Building a model is milliseconds; simulating it a few hundred times is not. That is why the screen can afford to be crude: rejecting a model costs one short simulation and saves several hundred long ones.

**One source, two artifacts.** Whatever is sampled must produce *both* the equation graph the encoder reads *and* the integrator that runs. Expand the card into SymPy once, canonicalise, then walk the same tree twice: walk 1 emits DAG nodes and edges, walk 2 emits the integrator. Write both walks yourself rather than using a converter, since walk 1 is needed regardless and having both traverse the identical tree in the identical order is exactly the invariant being protected. Add a round-trip assertion.

Handle the channels outside the scalar expression: compute $A_{c,i}$ once per step as a matrix-vector product, then evaluate the scalar part per region. The aggregation never has to be expressed in SymPy.

---

## 8. Implementation

The workload is unusual: state is tiny (68 regions × 1 to 6 variables), the time loop is long (900,000 steps at $dt = 1$ ms for a 15-minute scan), and there are many of them. **Parallelism is over parameter sets, not within a simulation.** A GPU wants 10⁵ to 10⁶ concurrent work items; one simulation gives on the order of 10². Batch a few thousand parameter sets.

| | |
|---|---|
| Framework | **JAX.** `lax.scan` for the time loop, `vmap` over parameter sets, the whole thing under one `jit` |
| Solver | Euler–Maruyama, **fixed $dt = 1$ ms**. Additive noise gives it strong order 1.0 and the Milstein correction vanishes, so there is no better solver to reach for |
| Precision | float32 |
| Output | reduce to FC and FCD **on the GPU**; returning raw BOLD is hundreds of MB per batch |
| Sampling | emit BOLD once per TR via a nested loop (outer over TRs, inner over steps), never every step |
| Backend | write walk 2 against a small op interface so JAX, NumPy and (later) PyTorch backends come from one tree walk. The NumPy backend gives GPU-free unit tests and a free cross-check |

**Why $dt = 1$ ms.** Sized against the fastest reference model: FIC's $\tau_I = 10$ ms gives a rate of 100 s⁻¹, so $dt\,\lambda = 0.1$, comfortably inside explicit Euler's accuracy regime. The published corpus uses 6 ms for training data and 0.5 ms for the final E/I computation; 6 ms puts FIC at $dt\,\lambda = 0.6$, which is validated but marginal, and with constants now spanning eight decades that margin is worth keeping.

Two consequences of a fixed step. **$dt$ now sets the upper bound on sampled rates**: explicit Euler is unstable above $dt\,\lvert\lambda\rvert = 2$, so anything faster than about 2000 s⁻¹ produces NaN and is rejected. The effective ceiling comes from $dt$ rather than from the prior, which means changing $dt$ later silently changes which models the corpus contains. And **cost**: 900,000 steps is six times the count at 6 ms, so roughly 20 seconds per batch of 1024 and under an hour for the full corpus. Not a constraint. If measurement says otherwise, 2 ms still leaves FIC at $dt\,\lambda = 0.2$.

**Why JAX rather than PyTorch.** The deciding factor is that 900,000 steps is a lot of sequential work. `lax.scan` compiles the entire time loop into one XLA computation, so the loop runs on-device. PyTorch has no equivalent: the loop stays in Python and every iteration dispatches its kernels from the CPU, which at 30 to 50 ops per step is several million launches per batch. `torch.compile` on the step body recovers much of that, and CUDA graphs recover the rest, but both are workarounds for something JAX does by default.

The simulator and the training code never share a process (the corpus is written to disk and read back), so using JAX here does not constrain the training stack. Keep the op interface anyway: it costs about thirty lines, and it means a PyTorch backend is a day's work rather than a rewrite if the situation changes.

**Do not use diffrax for the SDE.** Its `VirtualBrownianTree` exists to give reproducible Brownian paths under adaptive stepping, which this workload does not use, and it carries real overhead: a reported issue shows the Euler solver running roughly 200× slower than a hand-written loop on a trivial SDE. Fixed-step Euler–Maruyama inside a `lax.scan` is about twenty lines.

**The loop structure that matters.** Nest two scans: the inner one runs the integration steps within a TR, the outer one emits BOLD once per TR. Without this you materialise 900,000 timepoints per simulation and run out of memory long before anything else becomes a problem.

```python
def one_step(carry, _):
    x, k = carry
    k, sub = jax.random.split(k)
    x = x + dt * drift(x, params, C) + jnp.sqrt(dt) * sigma * jax.random.normal(sub, x.shape)
    return (x, k), None

def one_tr(carry, _):
    carry, _ = jax.lax.scan(one_step, carry, None, length=steps_per_tr)
    return carry, observe(carry[0])

_, bold = jax.lax.scan(one_tr, (x0, key), None, length=n_tr)
```

Split the PRNG key inside the scan rather than pre-generating noise; the full noise array does not fit.

**One JAX-specific cost to plan around.** Each distinct model triggers a recompile, so fifty models means fifty compilations. Run all parameter sets for one model in a few large batches rather than interleaving models, and keep the batch size fixed so shapes stay static.

**As built (2026-10).**

- **Step.** dt = 1 ms with float32 state.
- **Kahan compensation.** Euler increments of slow states sit near float32 resolution at small dt, so every update uses Kahan-compensated addition (`integrate.kahan_add`). Without it, MFM's test cost was 0.52 instead of the original's 0.43.
- **Balloon–Windkessel** is integrated as extra state, in deviations from rest.
- **Wong–Wang** $u/(1-e^{-u})$ is evaluated by a fused, stable op (`ops.wong_wang`), because the literal subtree is infinite in float32 for $|u| < 6\times10^{-8}$.
- **Hardware.** The corpus runs as **single-thread CPU jobs**, one job per chunk of candidates, by the project lead's choice: the allowance is 200 CPUs per user. The GPU path works but is not used for the corpus. The statistics are computed in the same process (`sim/summary.py`), so raw BOLD is never written.
- **Measured cost.** About 3.6 h per 500 candidates (screen plus 100 full-length draws of each kept model, including per-model compilation). That is about 7 CPU-hours per 1,000 candidates.

**A correctness reference worth using.** TVB's RateML compiles a declarative model description to CUDA and was validated at 68 nodes, with Montbrió already available in its XML. Not worth building on, but generating MFM, FIC and Montbrió through it and checking against your own compiler is much stronger evidence than your implementation agreeing with itself.

---

## 9. Corpus size

| | |
|---|---|
| Synthetic models | ~100,000 |
| Parameter sets per model | ~100 (first pass; more can be added per model later) |
| Connectome | one of the 64 HCP-YA training group SCs per model, drawn uniformly at random (seeded) |
| **Total simulations** | **~10,000,000** (train) + ~2,000,000 (val, test) |
| Statistics per simulation | `regional` (68 × 6), `pairwise` (2278, arctanh FC), `fcd_cdf` (100 levels), `fc_moments` (3), `diverged`, `flat`, `theta` (68 × 5, NaN-padded), `psi` (2, NaN-padded), all float32 |

**Splits and validity (decided 2026-10-10, project lead).** Three disjoint model populations, each a separate random stream of the sampler: **train** (100,000 models, SC from the 64 training groups), **val** and **test** (10,000 models each, SC from the 14 validation and 13 test groups). Each count is of **valid** models: a model passes the 4-draw screen (§6) *and* at least one of its 100 base parameter draws is neither diverged nor flat. A split's corpus is its first N valid models by candidate index, so it is reproducible at any size, and it grows by screening more indices; more draws per model are added by extension runs that skip the screen. Diverged: any state or recorded value non-finite or beyond ±10⁶ at any frame boundary. Flat: every region's temporal SD of the recorded signal at most 10⁻⁵ × max(1, |mean|). Statistics are stored in float32.

**Revised 2026-10-05** (project lead). The earlier plan was ~50 models × ~200 parameter sets × 4 SC bootstraps (~40,000 simulations). SC is no longer a per-simulation sampling axis. Each synthetic model is tied to one randomly chosen training group SC, so SC still varies across the corpus (architecture.md §7) without multiplying the simulation count. Only training-split groups are used, so validation and test SCs stay unseen in stage 1. First pass: 100 parameter sets per model (~6,500-9,700 CPU-hours, 1.3-2 days on 200 CPUs, by the single-CPU benchmark of 2026-10-05). The plan of ~1,000 per model can be reached by adding parameter sets to existing models, so the corpus format must allow appending. Each simulation stores its summary statistics (architecture.md §5.1) with the FCD CDF at 100 levels (data.md §8): about 11.6 KB per simulation, ~115 GB for 10⁷ simulations.

**Generation, as run (2026-10-10).** Seed 20261005 (`configs/corpus.yaml`). The streams are train 0, val 1 and test 2. Every random draw is keyed by (seed, stream, candidate index, purpose, draw), so any model or draw can be recomputed alone.

- **Valid rate.** Over the first 397,500 candidates, 15.6% are valid in every split: 16.2% pass the screen, and 0.6% pass but have no usable base draw. Within valid models, 14.7% of draws diverge and 0.8% are flat.
- **State count.** Valid models lean to small $V$: 57% have $V=1$, 38% $V=2$ and 5% $V=3$, against 30/45/15% when sampled. The project lead chose to proceed with this skew.
- **Ranges submitted.** Candidates 0–660,000 for train and 0–66,000 each for val and test, in 500-candidate jobs. At 15.6% that gives about 103,000 and 10,300 valid models, a margin of about 10 standard deviations for train and 5 for val and test.
- **Storage.** About 1.25 MB per valid model (100 draws), about 150 GB for all three splits.

**Files.** Each job writes `outputs/corpus/<split>/candidatesAAAAAAA-BBBBBBB_draws0-100.*` (`sim/corpus.py::chunk_paths`):

- `.npz` holds the per-draw arrays of the chunk's valid models, with a `model` index and a `draw` index.
- `.cards.jsonl` holds each valid model's card and SC group.
- `.log.jsonl` has one row per candidate: screen outcome, validity, $V$, $P$, $K$, DAG size and mean FC. These rows give the acceptance rate and mean FC that brief §8 asks to log.
- `.manifest.json` records the configs, seed, git commit, host and wall time.

A job that fails writes nothing, and resubmitting the same range recomputes it exactly. `scripts/corpus_status.py` reports progress, and `corpus.first_valid(split, n)` returns a split's corpus.

**Scaling up.** More models means screening more indices: `scripts/submit_corpus.py --start … --stop …`. More draws per model means extension runs (`--extend --first-draw 100 --n-param-sets 100`), which simulate draws 100.. of models already found valid, without re-screening. These are written as separate `_draws100-200` files.

Cheaper than it looks: stage 1 needs no empirical pairing, so one simulation is one training sample.

---

## 10. What to run, in order

**Step 0, hand-write the five.** Fill in the template's slots for Linear, MFM, FIC, Wilson–Cowan and Hopf, compile each, and check the simulator reproduces known behaviour and the published costs. **This is a gate.** If the template cannot reproduce the five exactly, nothing downstream means anything.

**Status (2026-10-10).** All seven reference cards compile, and each card's compiled right-hand side equals its published equations (`tests/test_compile.py`). The cost-reproduction gate (build step 6) is implemented but **deferred, not passed**: the project lead deferred it on 2026-10-05 and asked to proceed as if it had passed. Direct comparisons with the original simulations are in data.md §9.

Verification means matching *dynamics and cost values*, not the paper's constant tables. Nondimensionalisation and constant absorption mean the cards will not look character-for-character like the published equations.

Writing all five out is also how the subtler constraints surface. Hopf's $L$ has two distinct numbers rather than four; FIC's $w_{IE}$ is regional but not searched; Wilson–Cowan's threshold is not absorbable in its published form. None of those was visible before the substitution was done in full.

**Step 1, no synthetic data at all.** Train one shared backbone jointly on FIC, MFM and Hopf using the corpora you already have. Hold each out in turn and adapt on 0.1 / 1 / 5 / 10% of its corpus.

This costs almost no new simulation and it is the decisive experiment. If a shared backbone shows no leave-one-out transfer at 5% with three real models, 50 synthetic ones will not fix it.

**Step 2, generate and pretrain.** Only if step 1 is encouraging. 50 models, pretrain, then fine-tune to each of the five.

**Step 3, the control.** Template encoder zero-shot against a learned 128-dim model embedding fitted on ~50 simulations of the held-out model with the backbone frozen. If a fitted embedding matches the equation encoder, the equation encoder is not earning its complexity.

**Metric throughout:** simulations of the target model needed to reach a given optimization regret, pretrained versus from scratch. Not cost MSE.

---

## 11. Deliberately left out

Recorded so these read as decisions rather than oversights, and so they are easy to add back one at a time.

**Sampler restrictions** (the template supports these; hand-written cards may use them)

- **Filter pairs.** Critically damped $\{-1;\ a^2,\ 2a\}$ triples cannot arise from independent sampling: only $\zeta = 1$ gives a synaptic impulse response, and that has measure zero. So the corpus contains no second-order synapses, and **Jansen–Rit is template-reachable but not sampler-reachable**. A Jansen–Rit failure at test time would not distinguish "cannot read filter pairs" from "has never seen anything like this."
- **Coefficient tying.** Hopf's $L$ diagonal, its two channel gains, and $\sigma$ across variables. Generated Hopf-shaped models therefore carry more regional parameters than the published model, which is a diversity gain rather than a loss. The card schema keeps the ability to declare shared values so step 0 can reproduce Hopf exactly.
- **Implicit constraints** like FIC's analytically solved $w_{IE}$.
- **Zero-shifted transfer functions**, absorbed rather than sampled.
- **Non-one-hot channel sources** $m^c$, which Jansen–Rit needs to transmit a difference.

**Not in the template yet**

- Multiple nonlinear terms per variable, and conductance driving forces (Larter–Breakspear, Liley)
- Conduction delays, which also need a delay integrator (Robinson)
- Piecewise dynamics, which need a `Step` primitive (Epileptor)
- Circular state spaces (Kuramoto), not a slot value but a different state space
- Observation models expanded into the DAG, which would need a `Pow` primitive and unlocks regional hemodynamic parameters and EEG

**Corpus design**

- Spatially structured regional parameter maps (T1w/T2w, gradients). **The most likely of these to be needed early**: i.i.d. regional parameters produce FC dominated by SC. Watch whether mean off-diagonal FC tracks SC too closely across parameter draws
- Parameter-box calibration, cost-landscape screening, stratified sampling, fingerprint deduplication, tiered held-out design

**Encoder**

- The state-variable identifier, deferred with no forcing trigger yet
- Per-slot or tiered constant ranges, and any timescale screen. One flat log-uniform prior, one fixed $dt$
- *(No longer deferred: the log-magnitude constant encoding, reinstated because the eight-decade prior triggered it)*
