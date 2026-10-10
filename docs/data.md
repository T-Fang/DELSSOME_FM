# Empirical data: HCP-YA group-level SC, FC and FCD

**Status (2026-10-05).** Phase 0 is complete and its gate passes. Group membership is implemented in `src/delssome_fm/data/groups.py` and loading and averaging in `src/delssome_fm/data/empirical.py`. `tests/test_groups.py::test_all_reference_groups_reproduced_from_subject_data` recomputes all 91 reference groups within tolerance (§5).

**Updated 2026-10-10.** §6 and §7 record what build step 6 established about the original test data and simulation. §9 is the direct comparison with the original simulations. The open questions of §8 are now decided.

Path conventions. `HCP-YA/` means `/home/ftian/storage/projects/lifespan_EI/data/HCP-YA/` (`/home/ftian/storage` is a symlink to `/mnt/nas/CSC21/Yeolab/Users/ftian`). `tzeng:` means `/mnt/nas/CSC21/Yeolab/Users/tzeng/`, which Tianchu Zeng's scripts call `/home/tzeng/storage/`.

---

## 1. Summary

- There are **91 reference groups**: 64 train, 14 val and 13 test. Each group has a 68×68 SC, a 68×68 FC and a 10,000-bin FCD CDF.
- **Per-group membership is not saved anywhere, and it does not need to be.** The original grouping is deterministic: a sliding window of 50 subjects with stride 10, run inside contiguous splits of the 1029 subjects sorted by ascending ID. No RNG is involved.
- Recomputing every group from subject-level data reproduces the reference. The worst relative error per edge is 2.2e-16 for SC and 2.2e-11 for FC, and the FCD CDF matches exactly.

## 2. Inputs

| Data | Path | Per file | Notes |
|---|---|---|---|
| Subject order | `HCP-YA/pFIC_input/DK68/sub_list.txt` | 1029 IDs | Strictly ascending. Identical to `tzeng:Matlab/DELSSOME/HCP/general_matfiles/subject_1029.mat` and both copies of `HCP1029_sublist.txt` |
| Subject SC | `HCP-YA/SC/DK68/{id}.csv` | 68×68 | Raw streamline counts with a nonzero diagonal. 27 subjects have no SC: every field is empty and parses as NaN |
| Run FC | `HCP-YA/FC/DK68/{id}_bld00{k}.csv` | 68×68 | 3748 runs. `bld001..004` = REST1_LR, REST1_RL, REST2_LR, REST2_RL |
| Run FCD CDF | `HCP-YA/FCD_cdf/DK68/{id}_bld00{k}.csv` | 1×10000 | Cumulative counts, for the same runs as FC |
| Subject FC and FCD (run-averaged) | `HCP-YA/1029_sub/{FC,FCD_cdf}/DK68/{id}.csv` | | Not needed. Matches Tianchu's subject-level `.mat` to 1.1e-16 (FC) and 2.9e-11 counts (FCD) |
| **Reference group FC** | `HCP-YA/FC/DK68/group_dl_ds/{split}{g}.csv` | 68×68 | |
| **Reference group FCD CDF** | `HCP-YA/FCD_cdf/DK68/group_dl_ds/{split}{g}.csv` | 1×10000 | |
| **Reference group SC** | `HCP-YA/SC/DK68/group_dl_ds/{split}{g}.csv` | 68×68 | |
| Group myelin and RSFC gradient | `HCP-YA/{myelin,rsfc_gradient}/DK68/group_dl_ds/` | 1×68 | Unused for now. These are the natural input if spatially structured parameter maps (generation.md §11) are added |

`{split}{g}` runs from `train0` to `train63`, `val0` to `val13`, and `test0` to `test12`. Indices start at 0.

**Provenance.** The group CSVs were extracted by `HCP-YA/script/get_input.ipynb` (cell 4) from `tzeng:Matlab/DELSSOME/HCP/dl_grouped_mats/DK68/{train,val,test}.mat`. Those files hold only five variables: `fc_groups`, `fcd_cdf_groups`, `sc_groups`, `myelin_groups` and `rsfc_groups`. The per-run CSVs were copied by `HCP-YA/script/get_FC_FCD.py` from `tzeng:Matlab/DELSSOME/HCP/FC_FCD_CDF_DK68/{id}.mat`.

**Reading the CSVs.** Use `numpy.loadtxt`, or `pandas.read_csv(..., float_precision="round_trip")`. Both reproduce the `.mat` values bit for bit. The default pandas parser is off by up to 1.8e-15. That is harmless at the tolerances below, but there is no reason to accept it.

## 3. Conventions of the reference arrays

- **FC.** The diagonal is 0.9999999999999998, which is 1 − ε left over from the stable-atanh round trip (§4.2). FC is symmetric to 4e-16, but not bitwise.
- **SC.** The diagonal is 0 and the matrix is bitwise symmetric. Nonzero entries are log(mean nonzero streamline count) and range from about 0 to 10.5. About 57% of entries are nonzero.
- **FCD CDF.** Cumulative counts over 10,000 equal-width bins on [−1, 1], so each bin is 2e-4 wide. The last entry is always 624,403 = C(1118, 2): the number of upper-triangular FCD entries for 1118 windows. Divide by the last entry to get a CDF on [0, 1].

## 4. How the groups were built

### 4.1 Membership (brief §5.2, questions 2 to 4)

- **Is membership recorded?** Not per group. Per split, yes: `HCP-YA/pFIC_input/DK68/sub_list_{train,val,test}.txt`.
- **Deterministic or RNG-dependent?** Deterministic. The source is `tzeng:Matlab/DELSSOME/scripts/HCP_group_mats_for_dl_main_DK68.m`, which calls `HCP_group_mats_bootstrap.m` with `in_group_num = 50` and `stride = 10`. There is no RNG anywhere, so nothing needs porting.

Positions are 0-based in the ascending-ID list, and ranges are half-open:

| Split | Positions | MATLAB rows | Subjects | Groups | First and last ID |
|---|---|---|---|---|---|
| train | [0, 680) | 1–680 | 680 | 64 | 100206 … 459453 |
| val | [680, 860) | 681–860 | 180 | 14 | 461743 … 727553 |
| test | [860, 1029) | 861–1029 | 169 | 13 | 727654 … 996782 |

Within a split of *n* subjects, group *g* covers split positions [10*g*, min(10*g* + 50, *n*)), for *g* = 0 … ⌈(*n* − 50)/10⌉.

Consequences:

- **"Bootstrap" in the MATLAB file names means overlapping windows, not resampling.** Groups *g* and *g*+1 share 40 of their 50 subjects; groups *g* and *g*+5 share none. So the 64 training SCs are far fewer than 64 independent connectomes: only about 13 of them are pairwise disjoint. The corpus draws one group SC per synthetic model from its split's groups (generation.md §9), so this limits how much SC variety stage 1 sees.
- **test12 has 49 subjects** (positions 980–1028), because the final window is truncated at the end of the split. Every other group has 50.
- The splits are disjoint, so groups in different splits share no subjects.

**Do not derive groups from `HCP-YA/demogr/demogr.csv` or `HCP-YA/demogr/sub_list.txt`.** They contain the same 1029 IDs in a different order (sorted by age).

### 4.2 Averaging

| Level | FC | FCD CDF | SC |
|---|---|---|---|
| Run | Pearson correlation of the 68 DK ROI time series: 1200 frames at TR 0.72 s | Windows of 83 TRs at stride 1, giving 1118 windows. Pearson correlation between the windows' FC upper triangles, with **no arctanh**. The upper triangle of the 1118×1118 FCD matrix is binned into 10,000 bins on [−1, 1], then cumulatively summed | — |
| Subject | Fisher mean over the subject's runs | Mean over the subject's runs | One matrix per subject |
| Group | Fisher mean over subjects | Mean over subjects | `group_sc` (below) |

- **Fisher mean.** tanh(meanₖ stable_atanh(rₖ)). `stable_atanh` clips to [−1, 1] and maps ±∞ to atanh(±(1 − ε)) (`CBIG_StableAtanh`). Means skip missing runs and subjects (`CBIG_nanmean`).
- **`group_sc`** (`tzeng:Matlab/Utils/scripts/group_sc.m`). Drop subjects without SC. For each i ≠ j, if at least half of the remaining subjects have SCᵢⱼ ≠ 0, the entry is log(mean of the nonzero SCᵢⱼ); otherwise it is 0. The diagonal is 0.
- **Runs.** Runs with a frame count other than 1200 were discarded upstream (`HCP_Get_FC_FCD_DK68.m`). Of the 1029 subjects, 826 have 4 runs, 72 have 3, 97 have 2 and 34 have 1. Every subject has at least one.
- **Missing SC.** 27 subjects have no SC: 18 in train, 5 in val and 4 in test. Group SC is therefore averaged over 45 to 50 subjects, while FC and FCD always use the full group. The IDs are 109325, 113417, 114924, 116120, 121315, 121820, 128329, 150423, 159845, 169141, 171128, 171734, 186949, 190132, 201717, 209531, 212823, 239136, 552544, 613235, 623137, 662551, 689470, 734247, 822244, 901038 and 953764.

## 5. Verification

**How to run.** `pytest -m cluster` (about 9 minutes, almost all of it reading about 8,500 CSVs from the NAS), or `python scripts/verify_groups.py`. Off-cluster these tests skip. `pytest -m "not cluster"` runs the hand-computed unit tests of the window rule and the averaging functions.

The tolerances are those in brief §5.3, unchanged, and are fixed as constants in `src/delssome_fm/data/verify.py`. Relative error per edge is |x − y| / |y|. Where the reference is exactly 0 (SC zeros and the SC diagonal), the recomputed value must also be exactly 0. The FCD CDF is compared in stored count units, which is stricter than comparing normalised CDFs.

The test uses route B. Route A was checked during the investigation as a cross-check.

| Quantity | Tolerance | Route A: Tianchu's subject-level `.mat` → group | Route B (the test): per-run CSVs → subject → group | Worst entry (route B) |
|---|---|---|---|---|
| Group SC | rel < 1e-6 per edge | 2.2e-16 | 2.2e-16 | test6, edge (47, 63) |
| Group FC | rel < 1e-6 per edge | 1.4e-12 | 1.2e-11 | train59, edge (30, 63). The reference value is −7.7e-7, so the relative error is inflated by a near-zero denominator |
| Group FCD CDF | max abs < 1e-6 | 0 | 0 | Exact, in counts |

Route B also confirms the subject-level step (checked during the investigation). Run → subject Fisher means match `tzeng:.../general_matfiles/fc_DK68_1029.mat` to 3.3e-16 absolute, and FCD means match `fcd_cdf_DK68_1029.mat` exactly. The subject SC CSVs are bitwise equal to `sc_DK68_1029.mat`.

**Timestamps.** `HCP_group_mats_bootstrap.m` and `HCP_group_mats_for_dl_main_DK68.m` were last modified on 2024-04-18 at 16:53, a day after the `.mat` files were written (2024-04-17, 16:24 to 16:27). The scripts as they stand still reproduce the files exactly, so whatever changed did not affect the output.

## 6. A second group set: FIC_inv

The files `HCP-YA/{FC,SC,FCD_cdf}/DK68/FIC_inv/FIC_inv_{train,val,test}.csv` are single group averages over contiguous thirds of the test split:

| File | Positions | Subjects |
|---|---|---|
| `FIC_inv_train` | [860, 917) | 57 |
| `FIC_inv_val` | [917, 973) | 56 |
| `FIC_inv_test` | [973, 1029) | 56 |

They use the same averaging (`get_input.ipynb` cell 6; Tianchu's `scripts/run.m`). Recomputation matches: SC exactly, FC to ≤ 2.0e-14 relative error, and FCD to ≤ 5.6e-16 on the normalised CDF. This is the pFIC-style train/validation/test protocol.

**The published test costs were computed against `FIC_inv_test` (confirmed 2026-10-10).** The original test runs read `tzeng:Python/DELSSOME_plus/input/HCPYA/DK68/{SC_test.csv, FC_test.csv, FCD_CDF_test.mat}`. These equal `FIC_inv_test` up to the 5 significant digits of the CSVs: max abs difference 4.4e-4 for SC (max 10.5) and 5.0e-6 for FC; the normalised FCD CDFs agree to 8e-17. Build step 6 (`sim/reproduce.py::load_test_inputs`) reads Tianchu's files themselves, so that it uses exactly what the original used.

## 7. What the DELSSOME paper adds

From the paper and SI in `docs/` (Zeng, Tian et al., bioRxiv 2025.04.07.647497, version of 2026-03-24).

- **Groups.** §4.3 says participants were "repeatedly sampled" in groups of 50, giving 64/14/13 groups. The code (§4.1) shows what this means in practice: deterministic overlapping windows, not random draws.
- **SC is rescaled before simulation.** SI S3: group SC is computed as in §4.2 "with the maximum value normalized to 0.02". The stored reference SCs are *not* rescaled (their maximum is about 10.5), so the rescaling (SC × 0.02 / max SC) belongs to the simulation input, not to the loader. Confirmed in the simulation code (`sc_mat / torch.max(sc_mat) * 0.02` in each `CBIG_*_optimizer.py`). Ours is `sim/integrate.py::rescale_sc`, applied to every SC that is simulated, including the corpus SCs.
- **The FC correlation cost: the SI and the code disagree.** SI S2 says r is computed between *arctanh-transformed* FC upper triangles. The code that produced the published numbers (`tzeng:Python/DELSSOME_plus/scripts/utils/CBIG_pFIC_utils.py:204-231`, `FC_correlation_n_L1_cost`) correlates the **raw** upper triangles. To reproduce the published costs, use raw r. Kong 2021's pMFM code does use Fisher z, which may be where the SI wording came from.
- **The rest of the cost, as coded** (same file, lines 155-308):
  - d = |mean(FC_sim upper triangle) − mean(FC_emp upper triangle)|. This is a difference of means, not a mean absolute difference.
  - KS = max over the 10,000 bins of |CDF_sim − CDF_emp|, unsigned, with both CDFs normalised to end at 1.
  - Total cost = (1 − r) + d + KS.
  - Simulated FC is `corrcoef` on raw BOLD after dropping burn-in.
  - Simulated FCD uses the same 83-frame windows as the empirical pipeline, with `histc` into 10,000 bins on [−1, 1].
  - Over the 3 noise repeats, simulated FC is averaged arithmetically and FCD histograms are summed.
- **Simulation protocol of the original.**
  - Euler steps: dt = 6 ms (tzeng) or 5 ms (lifespan_EI) during CMA-ES, and 0.5 ms for the final test cost. Hopf uses 1 ms throughout.
  - Noise is σ·√dt·ξ, drawn independently for each state variable.
  - Each run starts with 5,000 warm-up steps, then burn-in (2.4 min, or 1.2 min for Hopf) and 14.4 min of simulation. BOLD is sampled every TR, so 1200 frames remain after burn-in is dropped.
  - FIC-inversion runs used the group SC of the 680 training subjects throughout, rescaled to SC × 0.02 / max.
- **Recorded costs on the HCP-YA FIC-inversion test group** (mean over 50 CMA-ES seeds, Euler arm). Source: `tzeng:Python/DELSSOME_plus/params/<model>_HCPYA/trial1/test/seed*/test_results.pth`, which matches `tzeng:Python/DELSSOME_plus/analysis/source_data/source_data.xlsx`. These are the numbers behind Figs 3c and 5b,e, and the targets for the build-step-6 gate.

  | Model | Mean 1 − r | Mean d | Mean KS | Mean total | Median total | Min total |
  |---|---|---|---|---|---|---|
  | FIC | 0.298 | 0.215 | 0.288 | 0.800 | 0.747 | 0.577 |
  | MFM | 0.391 | 0.095 | 0.418 | 0.904 | 0.776 | 0.389 |
  | Hopf | 0.318 | 0.027 | 0.159 | 0.504 | 0.470 | 0.379 |

  Best single seeds give a sharper target: MFM seed 30 has total 0.389 (0.230 / 0.027 / 0.132). FIC's best is seed 39 and Hopf's is seed 49; their saved parameters are in the same `test_results.pth` files.
- **Quirks in the original code that affect reproduction:**
  - Hopf's last BOLD frame is always 0, because it records on `(t+1) % t_inter` with no final write.
  - BOLD includes a factor of 100/ρ, which a code comment calls "a nonsense multiplication".
  - lifespan_EI's `Mfm2013.simulate` expects minutes, but its HCP-YA config gives seconds.

## 8. Decisions on the questions left open by Phase 0

All of these were open at the end of Phase 0 and are now settled.

1. **FCD resolution: 100 levels in the stage-1 corpus (decided 2026-10-05).** The corpus stores the simulated FCD CDF at **100 levels**: the CDF at FCD values −0.98, −0.96, …, 1.00, i.e. every 100th bin of the 10,000-bin CDF (`summary.fcd_levels`, stride 100). That makes the representation a CDF at fixed FCD values, not quantiles. Stage 2 must reduce empirical CDFs with the same function. Measured on the 91 empirical group CDFs, KS over these 100 levels underestimates the 10,000-bin KS by at most 0.0004 (mean 0.00007). Known cost of the even grid: the empirical FCD puts almost all its mass on [0.2, 1.0] (CDF < 0.001 below 0.21 in every group), so about 60 of the 100 levels sit where the empirical CDF is ~0. Costs of target-model simulations (build step 6, stage 2) are still computed at full resolution.
2. **Simulated scan length: as the data.** Simulations record 1200 frames at TR 0.72 s after discarding 200 frames (144 s) of burn-in (`configs/sim.yaml`), so the FCD has the same 1118 windows of 83 frames as the empirical pipeline. This replaces generation.md §8's 15-minute (1250-frame) sizing.
3. **No arctanh inside FCD.** Implemented: `summary.fcd_matrix` correlates raw windowed FC upper triangles, as the empirical pipeline does. arctanh is applied only to the pairwise FC targets (architecture.md §5.1). The original FC correlation cost does not use it either (§7).
4. **SC bootstraps: dropped as a sampling axis (decided 2026-10-05).** Each synthetic model is tied to one group SC, drawn uniformly (seeded) from its split's groups: train models from the 64 training groups, val and test models from the 14 and 13 val and test groups (generation.md §9). Overlap between groups (§4.1) is accepted.
5. **Cost-reproduction protocol (build step 6).** The published costs refer to the FIC_inv test group (§6). KS is max |ΔCDF|, unsigned, as in the Python code that produced the published numbers, not the signed one-sided max of Tianchu's `scripts/KS_distance.m`. The FC correlation uses raw r (§7). The gate itself is **deferred, not passed** (project lead, 2026-10-05); its criterion is fixed in `configs/reproduce.yaml`.

## 9. Our simulator against the original (2026-10-10)

Build step 6 is deferred, but two direct comparisons with the original DELSSOME have been run. Both use the 50 surviving test parameter sets per model (`params/<model>_HCPYA/trial1/test/seed1..50`) and the original test protocol (§7; `configs/reproduce_once.yaml`: dt 0.5 ms for MFM and FIC, 1 ms for Hopf; 3 simulations per evaluation, FC averaged and FCD histograms summed).

**The original code replays exactly.** `scripts/run_original_test.py` reruns Tianchu's own tester (torch, read-only imports) with the torch seed stored with each result. For all 150 (model, seed) pairs, the recomputed cost equals the recorded one to within 4e-11. It also saves the simulated FC and FCD, which the original runs did not keep. Outputs: `outputs/reproduce/original_sim/<model>_seed<k>.npz`.

**Recorded cost vs one evaluation of ours** (median over the 50 sets; `scripts/reproduce_costs.py --config configs/reproduce_once.yaml`, `scripts/plot_reproduction.py`):

| Model | 1 − r | d | KS | Total |
|---|---|---|---|---|
| FIC: recorded / ours | 0.276 / 0.272 | 0.189 / 0.195 | 0.223 / 0.209 | 0.747 / 0.730 |
| MFM: recorded / ours | 0.334 / 0.330 | 0.083 / 0.078 | 0.382 / 0.390 | 0.776 / 0.770 |
| Hopf: recorded / ours | 0.307 / 0.305 | 0.019 / 0.023 | 0.140 / 0.150 | 0.470 / 0.466 |

**Cost between the original's simulation and ours.** The same cost formula, with the original's simulated FC and FCD in place of the empirical data (`scripts/plot_comparison.py`). The noise floor is the same cost between two of our own evaluations with independent noise. Median over the 50 sets:

| Model | Total: original vs ours | Total: ours vs ours | KS: original vs ours | KS: ours vs ours |
|---|---|---|---|---|
| FIC | 0.146 | 0.129 | 0.076 | 0.067 |
| MFM | 0.129 | 0.121 | 0.090 | 0.073 |
| Hopf | 0.120 | 0.127 | 0.090 | 0.095 |

1 − r and d sit at the noise floor for all three models. KS is slightly above it for FIC and MFM. Five sets are outliers, 1 for FIC and 4 for MFM, with 1 − r ≈ 0.9–1.0 between the two implementations. Their FCs are uncorrelated, which suggests the two simulations settled into different dynamical regimes. These have not been investigated.

Figures are in `outputs/reproduce/comparison/`. `cost_tianchu_vs_ours.png` shows the cost plot. `matrices/<model>_seed<k>.png` shows one simulation's FC and FCD from each implementation, with the original colour tables (`assets/colormaps/`) and a colour range shared by the two panels of each row.

None of this is the gate. The gate needs 20 noise repeats per set (`configs/reproduce.yaml`) and has not been run.
