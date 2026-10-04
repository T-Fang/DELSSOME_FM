"""Group membership, the averaging functions, and the Phase 0 verification (brief §5.3).

The first half runs anywhere and checks against hand-computed values. The tests marked
`cluster` read the real data and are skipped when it is not reachable.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from conftest import data_available
from delssome_fm.config import SplitBounds
from delssome_fm.data.empirical import fisher_mean, group_sc, nan_mean, normalise_cdf, stable_atanh
from delssome_fm.data.groups import define_groups, groups_from_config, load_subject_order, window_bounds
from delssome_fm.data.verify import relative_error, verify_all_groups, worst_per_quantity

SPLITS = SplitBounds(train=(0, 680), val=(680, 860), test=(860, 1029))
EPS = 2.0 ** -52

# ---------------------------------------------------------------- membership


def test_window_bounds_reproduce_reference_group_counts():
    assert len(window_bounds(680, 50, 10)) == 64
    assert len(window_bounds(180, 50, 10)) == 14
    assert len(window_bounds(169, 50, 10)) == 13
    # MATLAB: for sub = 1:10:129 -> last start 121 (1-based), stop clipped at 169
    assert window_bounds(169, 50, 10)[-1] == (120, 169)
    assert window_bounds(180, 50, 10)[-1] == (130, 180)


def test_define_groups_layout():
    ids = list(range(100000, 101029))
    groups = define_groups(ids, SPLITS, size=50, stride=10)
    names = [g.name for g in groups]
    assert len(groups) == 91
    assert names[0] == "train0" and names[63] == "train63"
    assert names[64] == "val0" and names[-1] == "test12"
    by_name = {g.name: g for g in groups}
    assert by_name["train1"].subject_ids == tuple(ids[10:60])
    assert by_name["val0"].subject_ids == tuple(ids[680:730])
    assert by_name["test12"].subject_ids == tuple(ids[980:1029])
    assert all(len(g.subject_ids) == 50 for g in groups if g.name != "test12")
    # overlapping windows: neighbours share 40 subjects, groups 5 apart share none
    t = [set(by_name[f"train{i}"].subject_ids) for i in range(6)]
    assert len(t[0] & t[1]) == 40 and not t[0] & t[5]
    # splits never share subjects
    split_sets = {s: set().union(*(g.subject_ids for g in groups if g.split == s))
                  for s in ("train", "val", "test")}
    assert not split_sets["train"] & split_sets["val"]
    assert not split_sets["val"] & split_sets["test"]


def test_define_groups_rejects_bad_splits():
    ids = list(range(1029))
    with pytest.raises(ValueError, match="overlap"):
        define_groups(ids, SplitBounds(train=(0, 700), val=(680, 860), test=(860, 1029)), 50, 10)
    with pytest.raises(ValueError, match="not inside"):
        define_groups(ids, SplitBounds(train=(0, 680), val=(680, 860), test=(860, 1030)), 50, 10)


def test_load_subject_order_rejects_unsorted_and_wrong_count(tmp_path):
    path = tmp_path / "subjects.txt"
    path.write_text("100206\n100408\n100307\n")
    with pytest.raises(ValueError, match="strictly ascending"):
        load_subject_order(path, 3)
    with pytest.raises(ValueError, match="expected 4"):
        load_subject_order(path, 4)


# ---------------------------------------------------------------- averaging


def test_stable_atanh_maps_unit_correlation_to_finite_value():
    expected = 0.5 * math.log((2.0 - EPS) / EPS)  # atanh(1 - eps) written out
    z = stable_atanh(np.array([1.0, -1.0, 1.5, 0.0]))
    assert z[0] == pytest.approx(expected, rel=1e-12)
    assert z[1] == pytest.approx(-expected, rel=1e-12)
    assert z[2] == z[0]  # clipped to 1 first
    assert z[3] == 0.0


def test_fisher_mean_by_hand():
    def atanh(x):
        return 0.5 * math.log((1 + x) / (1 - x))

    def tanh(x):
        return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

    r = np.array([[0.5, 0.3], [-0.2, np.nan], [0.1, 0.6]])
    out = fisher_mean(r)
    assert out[0] == pytest.approx(tanh((atanh(0.5) + atanh(-0.2) + atanh(0.1)) / 3), rel=1e-12)
    assert out[1] == pytest.approx(tanh((atanh(0.3) + atanh(0.6)) / 2), rel=1e-12)  # NaN skipped


def test_fisher_mean_of_unit_diagonal_matches_reference_convention():
    assert fisher_mean(np.ones((4, 1)))[0] == 1.0 - EPS  # the reference FC diagonal


def test_nan_mean_raises_when_nothing_to_average():
    with pytest.raises(ValueError):
        nan_mean(np.array([[np.nan, 1.0], [np.nan, 3.0]]))


def test_group_sc_by_hand():
    def sc(e01, e02, e12, diag=7.0):
        return np.array([[diag, e01, e02], [e01, diag, e12], [e02, e12, diag]])

    subjects = np.stack([
        sc(2.0, 0.0, 4.0),
        sc(6.0, 0.0, 0.0),
        np.full((3, 3), np.nan),  # subject without SC: dropped, so 4 subjects count
        sc(4.0, 9.0, 0.0),
        sc(0.0, 3.0, 0.0),
    ])
    expected = np.array([
        [0.0, math.log(12 / 3), math.log(12 / 2)],  # (0,2): 2 of 4 nonzero meets "at least half"
        [math.log(4.0), 0.0, 0.0],                  # (1,2): 1 of 4 nonzero -> 0
        [math.log(6.0), 0.0, 0.0],
    ])
    np.testing.assert_allclose(group_sc(subjects), expected, rtol=1e-14, atol=0)


def test_group_sc_rejects_partly_missing_subject():
    subjects = np.ones((2, 3, 3))
    subjects[1, 0, 1] = np.nan
    with pytest.raises(ValueError, match="partly-NaN"):
        group_sc(subjects)


def test_normalise_cdf():
    np.testing.assert_array_equal(normalise_cdf(np.array([0.0, 1.0, 3.0, 4.0])),
                                  [0.0, 0.25, 0.75, 1.0])
    with pytest.raises(ValueError):
        normalise_cdf(np.array([0.0, 2.0, 1.0]))


def test_relative_error_treats_reference_zeros_strictly():
    err = relative_error(np.array([1.1, 0.0, 1e-9]), np.array([1.0, 0.0, 0.0]))
    assert err[0] == pytest.approx(0.1)
    assert err[1] == 0.0
    assert err[2] == np.inf


# ---------------------------------------------------------------- real data (Phase 0 gate)

# Provenance check only: the original list, on Tianchu Zeng's read-only storage.
TIANCHU_SUBJECT_LIST = Path(
    "/mnt/nas/CSC21/Yeolab/Users/tzeng/Matlab/DELSSOME/HCP/general_matfiles/subject_1029.mat")

needs_data = pytest.mark.skipif(not data_available(), reason="HCP-YA data not reachable")


@pytest.mark.cluster
@needs_data
def test_real_groups_have_expected_layout(data_cfg):
    groups = groups_from_config(data_cfg)
    assert len(groups) == 91
    by_name = {g.name: g for g in groups}
    assert by_name["train0"].subject_ids[0] == 100206
    assert by_name["test12"].subject_ids[-1] == 996782
    assert len(by_name["test12"].subject_ids) == 49


@pytest.mark.cluster
@needs_data
def test_subject_order_matches_tianchu_original(data_cfg):
    """Our copy of the subject list is the order the original grouping used."""
    if not TIANCHU_SUBJECT_LIST.is_file():
        pytest.skip(f"Tianchu's original list not mounted: {TIANCHU_SUBJECT_LIST}")
    original = loadmat(TIANCHU_SUBJECT_LIST)["subject_1029"].ravel().astype(int)
    assert tuple(original) == load_subject_order(data_cfg.subject_list, data_cfg.n_subjects)


@pytest.mark.cluster
@needs_data
def test_all_reference_groups_reproduced_from_subject_data(data_cfg):
    """Brief §5.3. Recomputes all 91 groups from per-run FC/FCD and subject SC (~5 min)."""
    results = verify_all_groups(data_cfg)
    assert len(results) == 91 * 3
    report = "\n".join(d.describe() for d in worst_per_quantity(results).values())
    failures = [d.describe() for d in results if not d.passed]
    assert not failures, "Phase 0 verification failed:\n" + "\n".join(failures) + "\n" + report
    print("\n" + report)
