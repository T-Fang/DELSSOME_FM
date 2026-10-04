# The spec layer: card → compiler → DAG and simulator

This directory turns a **model card** (a declarative description of a circuit model) into the
two things the rest of the project needs:

- the **equation DAG** that the template encoder reads (`nn/template_encoder.py`, Phase 4), and
- the **right-hand side** dx/dt that the integrator runs (`sim/integrate.py`, Phase 2).

Both come from **one** canonical SymPy tree, built in **one** `compile_card` call, with a
round-trip check on every compile. That is non-negotiable §2.1 of the brief. If the encoder
and the simulator ever disagreed about a model, every result would be silently wrong.

```
                        card.py                       compile.py
  ┌──────────────┐   load_card / validate   ┌────────────────────────────┐
  │ mfm.yaml     │ ───────────────────────► │ build_expressions          │  template of
  │ (ModelCard)  │                          │  (SymPy, evaluate=False)   │  generation.md §1
  └──────────────┘                          └─────────────┬──────────────┘
                                                          │ canonicalize
                                                          ▼
                                    derivs[v], edges[c], observable   (canonical trees)
                                         │                         │
                              walk A     │                         │    walk B
                         (_DagWalk)      ▼                         ▼  (_ProgramWalk)
                    ┌───────────────────────────┐   ┌────────────────────────────────┐
                    │ Dag (dag.py)              │   │ Program: straight-line code    │
                    │  nodes, edges, roots,     │   │ over the Ops interface (ops.py)│
                    │  Par node per parameter   │   │  -> CompiledModel.rhs(ops)     │
                    │  node_features -> M x 21  │   │  -> CompiledModel.observe(ops) │
                    └─────────────┬─────────────┘   └────────────────────────────────┘
                                  │ dag_to_sympy
                                  ▼
                  rebuilt trees == canonical trees ?   else CompileError
```

## 1. The card (`card.py`)

A `ModelCard` fills the slots of the template in `docs/generation.md` §1. For state variable
v of region i:

```
dx^v/dt = -Σ_u L[v,u] x^u + β_v + γ_v g_v(x) Φ_v(u^v) + Σ_{c: v∈D_c} A_c + σ_v ν^v
u^v     =  Σ_u w[v,u] x^u + I_v + Σ_{c: v∈U_c} A_c
A_c,i   =  G_c Σ_j C_ij ( f_c(p^c_j) − δ_c f_c(p^c_i) ),   p^c = Σ_u m^c_u x^u
```

| Card field | Template slot | Notes |
|---|---|---|
| `linear` (V×V) | L | enters as −Σ L x. The sampler always draws the diagonal; hand cards may omit it (MPR, Jansen–Rit) |
| `bias` | β_v | |
| `nonlinear[v]` | γ_v, g_v, Φ_v, w[v,:], I_v | `None` = no nonlinear term |
| `noise` | σ_v | `None` = noiseless variable |
| `channels` | f_c, m^c, δ_c, G_c, U_c, D_c | declared once, injected by variable index. `source` is the weight vector m^c: one-hot from the sampler, weighted in hand cards (Jansen–Rit sends y1 − y2) |
| `observable` | o_i, observation model | x^q or x^q − x^q′; Balloon–Windkessel or direct |
| `parameters` | which coefficients are free | regional ones are the P columns of Θ, global ones the K entries of Ψ |

Every coefficient slot holds a `Coef`: either a **constant** (`10.0`) or **scale × a named
parameter** (`{param: w, scale: 10.848222}`). Several slots may name the same parameter. That
is how coefficients are tied: Hopf's `a` appears on both diagonal entries of L.

Rules checked when a card is constructed (`CardError` on violation):

- shapes, indices and names are valid;
- **R2**: an identity transfer with gate 1 is affine, so it is forbidden (omit the term);
- **R3**: the first nonzero quadratic-gate weight must be 1;
- every declared parameter is used. Channel gains use global parameters, and every other slot
  uses regional ones;
- channels must differ and be injected somewhere. `into_input` needs a nonlinear term.

The R1 eigenvalue check on L belongs to the sampler (`sampler.py`, Phase 3), because hand-written
cards may carry parameters in L (Hopf).

**YAML.** `load_card`, `card_to_yaml` and `card_from_yaml` round-trip a card exactly. Unknown
or missing keys raise. Note that YAML reads `1e-4` as a *string*; write `1.0e-4`.

## 2. Canonicalisation (`compile.py`)

`build_expressions` writes the template with `evaluate=False`, so SymPy does **not** rewrite
it. Left alone, SymPy distributes numbers over sums (γ(1−S) → γ − γS) and pulls constants out
of `exp`, both of which destroy the template's structure. `canonicalize` then applies exactly
the normalisation in the brief, and nothing more:

- **constant folding:** the numbers among one sum's (product's) operands become one constant.
  Integral values become Integers, so `1.0·x` → `x`;
- **removal of zeros:** `x + 0 → x`, `0·x → 0`, `1·x → x`;
- **term ordering:** operands sorted by SymPy's `default_sort_key`;
- **one form of negation:** `Mul(−1, X)` with a single X. This is the DAG's `Neg`.

Products are not distributed and nested products are not flattened. `γ·g·Φ(u)` stays one
product with Φ as a child, and the `u` inside Wong–Wang's `u / (1 − exp(−u))` is one shared
node. `canonicalize` is idempotent; the round trip relies on that.

Transfer functions are written in primitives (generation.md §3: canonical, no constants):

| Φ | Expression | DAG |
|---|---|---|
| identity | u | u |
| logistic | 1 / (1 + e^{−u}) | `Inv(Add(1, Exp(Neg u)))` |
| Wong–Wang | u / (1 − e^{−u}) | `Mul(u, Inv(Add(1, Neg(Exp(Neg u)))))` |

## 3. The DAG (`dag.py`)

15 node types (fixed by the project lead on 2026-10-05):

| # | Type | | # | Type | |
|---|---|---|---|---|---|
| 1 | Add | n-ary sum | 9 | Const | numeric constant |
| 2 | Mul | n-ary product | 10 | Par | free parameter |
| 3 | Neg | Add(a, Neg b) is subtraction | 11 | Var | state variable, self or neighbour |
| 4 | Inv | Mul(a, Inv b) is division | 12 | EdgeAttr | C_ij, only under Agg |
| 5 | Exp | | 13 | Noise | ν^v_i |
| 6 | Log | reserved, unreachable now | 14 | Deriv | root, one per state variable |
| 7 | Sq | square | 15 | Out | root, the observable |
| 8 | Agg | Σ_j over one channel's edge expression | | | |

Node features, `node_features(dag)` → (M, 21):

| Columns | Content | Nonzero on |
|---|---|---|
| 1–15 | type one-hot | all |
| 16 | sign(c) | Const |
| 17 | clip(log10\|c\|, −4, 4) | Const |
| 18 | +1 regional, −1 global | Par |
| 19 | +1 self, −1 neighbour | Var |
| 20–21 | Balloon–Windkessel, direct | Out |

This replaces architecture.md §1's "1 parameter index, 1 scope tag". Which `Par` node is which
parameter is not a feature. It is `Dag.regional_par_nodes` / `global_par_nodes`, in Θ and Ψ
order, which the encoder needs for the κ_p and λ_κ readouts.

**Sharing.** Identical subexpressions are one node, so the "graph" is a DAG. The exception is
`Const`: each occurrence gets its own node, so that unrelated subtrees are not joined through a
shared literal such as 1. `EdgeAttr` and neighbour `Var` nodes are per channel (one neighbour `Var` per variable with nonzero m^c_u).

**Agg.** A channel appears in the per-region equations as one `Agg` node. Its child is the
edge expression `G_c · C_ij · (f_c(p^c_j) − δ_c f_c(p^c_i))`, so the encoder sees the gain, whether
coupling is diffusive, and f_c. The Agg depth (`depth_from_roots`) is 2 when the channel is injected at the
derivative (Linear, Hopf, MPR, Jansen–Rit) and 5 for MFM and FIC, as architecture.md §2 says.
Wilson–Cowan's is 8, because the logistic expands to `Inv(Add(1, Exp(Neg u)))`.

## 4. The right-hand side (walk B, `ops.py`)

Walk B emits a straight-line program over the `Ops` protocol, with one method per primitive.
`CompiledModel.rhs(ops)` returns `f(x, nu, theta, psi, sc) -> dx/dt` with shapes (N, V),
(N, V), (N, P), (K,), (N, N).

- **Noise is an input.** With ν = ξ/√dt and ξ standard normal, `x + dt·f(x, ν)` is *exactly* an
  Euler–Maruyama step. With ν = 0 you get the deterministic drift. The compiler checks that
  noise enters additively.
- **Channels** are computed once per step outside the per-region program as
  `G_c · (C @ f_c(p^c) − δ_c · rowsum(C) · f_c(p^c))`. That is algebraically the Agg sum, and
  `tests/test_compile.py` checks it against an explicit Σ_j of the edge expression.
- `NumpyOps` (float64) is for GPU-free tests; `JaxOps` (float32) is for simulation under
  `jit`/`vmap`.

## 5. Reference cards (`reference/`)

| Card | V | Regional (Θ columns) | Global | Observable | Card vs published parameters |
|---|---|---|---|---|---|
| `linear` | 1 | inv_tau, sigma | G | x, BW | no published fit |
| `mfm` | 1 | w, I, sigma | G | S, BW | **I_card = I − b/a**; the rest are equal |
| `fic` | 2 | wEE, wEI, sigma, **wIE** | G | S_E, BW | equal; wIE is solved by FIC, not searched (P = 4) |
| `wilson_cowan` | 2 | c1, P, sigma | c5 | E, BW | **P_card = P − θ_e**; illustrative constants |
| `hopf` | 2 | a, omega, sigma | g | x, direct | equal |
| `mpr` | 2 | eta, J, sigma | G | r, BW | equal; no published fit |
| `jansen_rit` | 6 | p, sigma | G | y1 − y2, BW | equal; **linear** y1 − y2 coupling (see below); no published fit |

The template has no constant slot inside u apart from the drive, so a published threshold is
absorbed into the drive parameter (generation.md §3). That is why two cards carry a shifted
parameter. Each YAML file documents its mapping in its header. The original DELSSOME only
implemented FIC, MFM and Hopf, so the other four cannot be checked against published costs.

**Jansen–Rit's coupling.** The usual sigmoidal network coupling S(y1_j − y2_j) puts the
threshold v0 inside the sigmoid, and the template has no constant slot inside f_c (only m^c,
which absorbs the slope). The card therefore couples y1 − y2 *linearly* into the excitatory
input. The single-region model is exact. Supporting the sigmoidal form would need an offset
in p^c, which would be a template change.

## 6. Extending the template

The design keeps every extension a matter of lengthening a list (generation.md §2):

- **A new transfer function** (e.g. softplus): add its name to `card.TRANSFERS`, write it in
  primitives in `compile.transfer`, and add a test against the published formula. If it needs
  a new primitive (Pow, Step), that changes the 15-type vocabulary and every trained encoder,
  so ask first.
- **A new gate:** add to `card.GATES`, validate it in `card._check_nonlinear`, and build it in
  `compile._gate`.
- **A new card:** write the YAML, compile it, and add its published equations to the
  `PUBLISHED` table in `tests/test_compile.py`.

## 7. Tests (`tests/test_compile.py`)

- All seven cards compile, round-trip through YAML, and have the expected V, P and K.
- **Each card's compiled rhs equals its published equations**, written out by hand in NumPy
  with the published constants (rtol 1e-11). This is the check that the absorbed constants
  are right.
- The rhs equals SymPy's `lambdify` of the canonical trees with an explicit neighbour sum,
  which exercises a different code path for walk B and the channels.
- `JaxOps` under `jit` matches `NumpyOps`.
- The 21-column features are right, Par nodes follow Θ/Ψ order, and no zero constant reaches
  the DAG.
- The round trip catches a tampered DAG; `canonicalize` is idempotent; Wong–Wang's u is
  shared.
- Card rules (R2, R3, scopes, unused parameters) raise.
