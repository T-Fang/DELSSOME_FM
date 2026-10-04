# Empirical data: HCP-YA group-level SC, FC and FCD

**Status (2026-10-05).** Phase 0 is complete and its gate passes. Group membership is implemented in `src/delssome_fm/data/groups.py` and loading and averaging in `src/delssome_fm/data/empirical.py`. `tests/test_groups.py::test_all_reference_groups_reproduced_from_subject_data` recomputes all 91 reference groups within tolerance (§5).

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

- **"Bootstrap" in the MATLAB file names means overlapping windows, not resampling.** Groups *g* and *g*+1 share 40 of their 50 subjects; groups *g* and *g*+5 share none. This matters when choosing the "SC bootstraps" for the corpus (generation.md §9): four adjacent groups are nearly the same connectome.
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

They use the same averaging (`get_input.ipynb` cell 6; Tianchu's `scripts/run.m`). Recomputation matches: SC exactly, FC to ≤ 2.0e-14 relative error, and FCD to ≤ 5.6e-16 on the normalised CDF. This is the pFIC-style train/validation/test protocol, and probably what the published FIC costs were computed against. Confirm this at build step 6.

## 7. Open questions for later phases

None of these block Phase 0. They are recorded here so they are not lost.

1. **FCD resolution (Phase 2).** The data has 10,000 bins, while architecture.md §1 and §5.1 use 100 fixed levels. The reduction needs choosing, and it must be applied identically to empirical and simulated CDFs. There is also an inconsistency in architecture.md:
   - §5.1 says "100 fixed *probability* levels", which describes quantiles.
   - The cumulative-softmax head (§5.2) and the pointwise KS (§6.5) imply a CDF evaluated at 100 fixed FCD values.
   - KS over 100 levels only approximates the 10,000-bin KS of the original cost.
2. **Simulated scan length (Phase 2).** The empirical FCD is defined on 1200 frames at TR 0.72 s (864 s), which gives 1118 windows of 83 TRs. generation.md §8 sizes the simulator for a 15-minute scan (900 s = 1250 frames). Brief §7.3 requires the same TR, window and stride. Decide whether simulations also use 1200 frames, so that the CDF's sampling noise matches the data.
3. **No arctanh inside FCD.** The empirical pipeline correlates raw windowed FC. The "same arctanh" in architecture.md §5.1 refers to the pairwise FC targets. `summary.py` must not apply arctanh inside the FCD computation.
4. **SC bootstraps (Phase 3).** Adjacent groups overlap by 80% (§4.1). Choose groups at least 5 apart to get distinct connectomes, and decide which split they come from.
5. **Cost-reproduction gate (build step 6).** Decide which group set the published costs refer to (FIC_inv test, or group_dl_ds) and which KS definition applies. Tianchu's `scripts/KS_distance.m` takes a signed, one-sided max rather than max |·|.
