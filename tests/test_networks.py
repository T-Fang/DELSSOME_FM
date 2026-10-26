"""data/networks.py: the Yeo-7 assignment and network FC computed from region FC and SDs
equals the FC of the network-averaged time series."""

import numpy as np

from delssome_fm.data.networks import (NETWORK_PAIRS, YEO7_NAMES, network_fc, stable_arctanh,
                                       yeo7_assignment)


def test_assignment_covers_all_networks_and_mirrors_the_hemispheres():
    networks = yeo7_assignment()
    assert networks.shape == (68,) and set(networks) == set(range(7))
    assert len(YEO7_NAMES) == 7 and len(NETWORK_PAIRS) == 21
    assert (networks[:34] == networks[34:]).sum() == 30      # 4 homologues differ (2026-10-26)


def test_network_fc_equals_fc_of_averaged_time_series():
    rng = np.random.default_rng(0)
    networks = yeo7_assignment()
    T, B = 500, 3
    fc_upper, sd, expected = [], [], []
    i, j = np.triu_indices(68, 1)
    for _ in range(B):
        shared = rng.standard_normal((T, 7))[:, networks]          # network-shared signal
        x = (shared + rng.standard_normal((T, 68))) * 10.0 ** rng.uniform(-3, 3, 68) + 5.0
        fc_upper.append(np.corrcoef(x.T)[i, j])
        sd.append(x.std(axis=0))
        avg = np.stack([x[:, networks == n].mean(axis=1) for n in range(7)], axis=1)
        c = np.corrcoef(avg.T)
        expected.append([c[a, b] for a, b in NETWORK_PAIRS])
    got = network_fc(np.array(fc_upper), np.array(sd), networks)
    np.testing.assert_allclose(got, np.array(expected), atol=1e-9)


def test_stable_arctanh_is_finite_at_one():
    assert np.isfinite(stable_arctanh(np.array([1.0, -1.0, 1.0000001]))).all()
    np.testing.assert_allclose(stable_arctanh(np.array([0.5])), np.arctanh(0.5))


def test_network_fc_survives_sds_that_would_overflow():
    networks = yeo7_assignment()
    rng = np.random.default_rng(1)
    fc = rng.uniform(-0.1, 0.1, (2, 2278))
    sd = 10.0 ** rng.uniform(150, 300, (2, 68))      # squares overflow float64
    r = network_fc(fc, sd, networks)
    assert np.isfinite(r).all() and np.all(np.abs(r) <= 1)
