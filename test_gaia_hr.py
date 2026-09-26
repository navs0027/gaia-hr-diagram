"""
Tests for gaia_hr_diagram.py. They need no internet: the physics is checked
against hand-calculable cases, and the plotting pipeline runs on a *synthetic*
star catalogue whose true absolute magnitudes are known in advance.

Run with either:
    python test_gaia_hr.py
    python -m pytest
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import gaia_hr_diagram as hr


# ---------------------------------------------------------------- physics
def test_absolute_magnitude_hand_cases():
    # A star at exactly 10 pc (parallax 100 mas) has M = m by definition.
    assert np.isclose(hr.absolute_magnitude(7.0, 100.0), 7.0)
    # At 100 pc (10 mas) it looks 5 magnitudes fainter than it is.
    assert np.isclose(hr.absolute_magnitude(7.0, 10.0), 2.0)
    # At 1 kpc (1 mas), 10 magnitudes fainter.
    assert np.isclose(hr.absolute_magnitude(12.0, 1.0), 2.0)


def test_absolute_magnitude_accepts_arrays():
    m = hr.absolute_magnitude(np.array([7.0, 7.0]), np.array([100.0, 10.0]))
    assert np.allclose(m, [7.0, 2.0])


def test_absolute_magnitude_error_at_10_sigma():
    # parallax_over_error = 10  ->  sigma_M = 2.1715 * 0.1 = 0.217 mag
    assert np.isclose(hr.ABS_MAG_ERR_FACTOR * 0.1, 0.2171, atol=1e-4)


# ------------------------------------------------------ astrometric cut
def _row(g, unit_weight_error, nu=100):
    # sqrt(chi2 / (nu - 5)) = unit_weight_error
    return dict(phot_g_mean_mag=g, astrometric_n_good_obs_al=nu,
                astrometric_chi2_al=unit_weight_error ** 2 * (nu - 5))


def test_astrometric_cut_thresholds():
    rows = [
        _row(15.0, 2.0),   # bright: limit = 1.2*exp(0.9) = 2.95 -> pass
        _row(15.0, 3.5),   # bright but too noisy               -> fail
        _row(20.0, 1.1),   # faint: limit = 1.2                 -> pass
        _row(20.0, 1.3),   # faint and above 1.2                -> fail
    ]
    keep = hr.astrometric_cut(pd.DataFrame(rows)).tolist()
    assert keep == [True, False, True, False]


def test_astrometric_cut_handles_bad_rows():
    df = pd.DataFrame([_row(15.0, 1.0, nu=5),          # nu - 5 = 0 -> division by zero
                       dict(phot_g_mean_mag=15.0, astrometric_n_good_obs_al=np.nan,
                            astrometric_chi2_al=np.nan)])
    assert hr.astrometric_cut(df).tolist() == [False, False]


# ------------------------------------------------------------------ query
def test_query_contents():
    q = hr.build_query(100.0)
    assert "FROM gaiadr3.gaia_source" in q
    assert "parallax > 10.0000" in q            # 100 pc  <->  10 mas
    assert "parallax_over_error > 10" in q
    assert "visibility_periods_used > 8" in q
    assert "phot_g_mean_flux_over_error > 50" in q
    assert "phot_bp_mean_flux_over_error > 20" in q
    assert "phot_rp_mean_flux_over_error > 20" in q
    assert "1.0 + 0.015 * POWER(bp_rp, 2)" in q
    assert "1.3 + 0.06 * POWER(bp_rp, 2)" in q
    assert "parallax > 5.0000" in hr.build_query(200.0)


# --------------------------------------------- waiting for the archive job
class FakeJob:
    """Stands in for an astroquery async job: reports a scripted list of phases."""

    def __init__(self, phases):
        self.phases, self.calls = list(phases), 0

    def get_phase(self, update=False):
        item = self.phases[min(self.calls, len(self.phases) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    """Time that only moves when sleep() is called, so tests need no real waiting."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _wait(job, timeout_s=1200, lines=None):
    clock = FakeClock()
    lines = [] if lines is None else lines
    hr.wait_for_job(job, timeout_s=timeout_s, poll_s=10, sleep=clock.sleep,
                    clock=clock.now, say=lines.append)
    return lines


def test_wait_for_job_reports_progress_then_returns():
    lines = _wait(FakeJob(["QUEUED", "QUEUED", "EXECUTING", "COMPLETED"]))
    text = "\n".join(lines)
    assert "QUEUED" in text and "EXECUTING" in text and "COMPLETED" in text


def test_wait_for_job_raises_on_archive_error():
    try:
        _wait(FakeJob(["EXECUTING", "ERROR"]))
    except RuntimeError as exc:
        assert "ERROR" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_wait_for_job_times_out():
    try:
        _wait(FakeJob(["QUEUED"]), timeout_s=300)
    except TimeoutError as exc:
        assert "QUEUED" in str(exc)
    else:
        raise AssertionError("expected TimeoutError")


def test_wait_for_job_survives_brief_network_blips():
    blip = ConnectionError("blip")
    lines = _wait(FakeJob([blip, blip, "EXECUTING", "COMPLETED"]))   # 2 failures: fine
    assert any("no answer from the archive" in line for line in lines)
    try:
        _wait(FakeJob([blip] * 10))                              # 5 in a row: give up
    except RuntimeError as exc:
        assert "lost contact" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_fetch_from_gaia_uses_background_job_and_returns_dataframe():
    """Wire the whole fetch path to a fake archive (no internet needed)."""
    import sys
    import types
    from unittest import mock

    from astropy.table import Table

    import socket

    seen = {}
    timeout_before = socket.getdefaulttimeout()

    class FakeArchiveJob(FakeJob):
        def get_results(self):
            return Table({c: [1.0, 2.0] for c in hr.COLUMNS})

    class FakeGaia:
        @staticmethod
        def launch_job_async(query, background=False, **kwargs):
            seen["query"], seen["background"] = query, background
            seen["socket_timeout"] = socket.getdefaulttimeout()
            return FakeArchiveJob(["EXECUTING", "COMPLETED"])

    fake_module = types.ModuleType("astroquery.gaia")
    fake_module.Gaia = FakeGaia
    with mock.patch.dict(sys.modules, {"astroquery.gaia": fake_module}), \
            mock.patch.object(hr.time, "sleep", lambda s: None):
        df = hr.fetch_from_gaia(100.0)

    assert seen["background"] is True
    assert "FROM gaiadr3.gaia_source" in seen["query"]
    assert seen["socket_timeout"] == hr.HTTP_TIMEOUT_SECONDS      # set during the fetch...
    assert socket.getdefaulttimeout() == timeout_before           # ...and restored after
    assert list(df.columns) == hr.COLUMNS and len(df) == 2


def test_timeout_option_is_wired_through_to_the_wait():
    import sys
    import types
    from unittest import mock

    assert hr.parse_args([]).timeout == hr.JOB_TIMEOUT_MINUTES == 60
    assert hr.parse_args(["--timeout", "5"]).timeout == 5

    class StuckGaia:
        @staticmethod
        def launch_job_async(query, background=False, **kwargs):
            return FakeJob(["QUEUED"])          # never finishes

    fake_module = types.ModuleType("astroquery.gaia")
    fake_module.Gaia = StuckGaia
    with mock.patch.dict(sys.modules, {"astroquery.gaia": fake_module}), \
            mock.patch.object(hr.time, "sleep", lambda s: None):
        try:
            hr.fetch_from_gaia(100.0, timeout_minutes=0.001)   # 0.06 seconds
        except SystemExit as exc:
            assert "gave up" in str(exc) and "QUEUED" in str(exc)
        else:
            raise AssertionError("expected a clean exit after the timeout")


def _fake_pyvo(job, hardlimit=1000):
    """A stand-in for the pyvo module: one TAP service that hands back `job`."""
    import types

    class FakeService:
        def __init__(self, url):
            self.url, self.submitted = url, {}

        @property
        def hardlimit(self):
            if hardlimit is None:
                raise RuntimeError("not exposed")
            return hardlimit

        def submit_job(self, query, maxrec=None):
            self.submitted = dict(query=query, maxrec=maxrec)
            FakeService.last = self
            return job

    dal = types.ModuleType("pyvo.dal")
    dal.TAPService = FakeService
    dal.DALOverflowWarning = type("DALOverflowWarning", (UserWarning,), {})
    module = types.ModuleType("pyvo")
    module.dal = dal
    return module, dal, FakeService


class FakePyvoJob:
    def __init__(self, phases, rows=2, overflow_warning=None):
        from astropy.table import Table   # imported here, not inside the warning-capturing block

        self.phases, self.calls, self.job_id = list(phases), 0, "abc123"
        self.table = Table({c: [1.0] * rows for c in hr.COLUMNS})
        self.overflow_warning = overflow_warning

    @property
    def phase(self):
        item = self.phases[min(self.calls, len(self.phases) - 1)]
        self.calls += 1
        return item

    def run(self):
        return self

    def raise_if_error(self):
        pass

    def fetch_result(self):
        import warnings

        if self.overflow_warning is not None:
            warnings.warn("Partial result set", self.overflow_warning)
        table = self.table
        return type("Result", (), {"to_table": lambda self_: table})()


def test_fetch_from_ari_returns_dataframe():
    import sys
    from unittest import mock

    job = FakePyvoJob(["EXECUTING", "COMPLETED"])
    module, dal, service_cls = _fake_pyvo(job)
    with mock.patch.dict(sys.modules, {"pyvo": module, "pyvo.dal": dal}), \
            mock.patch.object(hr.time, "sleep", lambda s: None):
        df = hr.fetch_from_ari(100.0)
    assert list(df.columns) == hr.COLUMNS and len(df) == 2
    sent = service_cls.last.submitted
    assert sent["maxrec"] == 1000 and "FROM gaiadr3.gaia_source" in sent["query"]
    assert service_cls.last.url == hr.ARI_TAP_URL


def test_fetch_from_ari_refuses_a_truncated_result():
    import sys
    from unittest import mock

    for case in ("exactly_at_the_row_limit", "overflow_warning"):
        job = FakePyvoJob(["COMPLETED"], rows=1000 if case == "exactly_at_the_row_limit" else 5)
        module, dal, _ = _fake_pyvo(job, hardlimit=1000)
        if case == "overflow_warning":
            job.overflow_warning = dal.DALOverflowWarning
        with mock.patch.dict(sys.modules, {"pyvo": module, "pyvo.dal": dal}), \
                mock.patch.object(hr.time, "sleep", lambda s: None):
            try:
                hr.fetch_from_ari(100.0)
            except SystemExit as exc:
                assert "cut the result off" in str(exc), case
            else:
                raise AssertionError(f"expected a truncated result to be refused ({case})")


def test_fetch_from_ari_falls_back_when_service_states_no_limit():
    import sys
    from unittest import mock

    job = FakePyvoJob(["COMPLETED"])
    module, dal, service_cls = _fake_pyvo(job, hardlimit=None)
    with mock.patch.dict(sys.modules, {"pyvo": module, "pyvo.dal": dal}), \
            mock.patch.object(hr.time, "sleep", lambda s: None):
        hr.fetch_from_ari(100.0)
    assert service_cls.last.submitted["maxrec"] == hr.FALLBACK_MAXREC


def test_load_data_downloads_once_then_uses_the_cache():
    import os
    from unittest import mock

    calls = []

    def fake_fetch(max_distance_pc, timeout_minutes):
        calls.append(timeout_minutes)
        return make_synthetic_catalogue(n=100).drop(columns=["true_abs_g"])

    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            with mock.patch.object(hr, "fetch_from_gaia", fake_fetch):
                first = hr.load_data(100.0, None, refresh=False, timeout_minutes=7)
                assert (Path("data") / "gaia_dr3_within_100pc_esa.csv").exists()
                second = hr.load_data(100.0, None, refresh=False, timeout_minutes=7)
                assert calls == [7]                       # second call came from the cache
                assert len(first) == len(second) == 100
                hr.load_data(100.0, None, refresh=True, timeout_minutes=7)
                assert calls == [7, 7]                    # --refresh queries again
            with mock.patch.object(hr, "fetch_from_ari", fake_fetch):
                hr.load_data(100.0, None, refresh=False, timeout_minutes=9, source="ari")
                assert (Path("data") / "gaia_dr3_within_100pc_ari.csv").exists()
                assert calls[-1] == 9
        finally:
            os.chdir(old_cwd)


def test_source_option():
    assert hr.parse_args([]).source == "esa"
    assert hr.parse_args(["--source", "ari"]).source == "ari"


def test_default_socket_timeout_stops_a_silent_server():
    """
    Reproduces the hang: a server that accepts the connection but never answers.
    Without a default timeout http.client waits forever (what astroquery did);
    with one it raises, which wait_for_job() can retry.
    """
    import http.client
    import socket
    import time as _time

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)                       # accepts the handshake, never replies
    port = server.getsockname()[1]
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(1.0)
    try:
        start = _time.monotonic()
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/phase")
        try:
            conn.getresponse()
        except TimeoutError:
            pass
        else:
            raise AssertionError("expected a timeout")
        assert _time.monotonic() - start < 5
    finally:
        socket.setdefaulttimeout(previous)
        server.close()


# ----------------------------------------------- synthetic end-to-end run
def make_synthetic_catalogue(n=60000, max_pc=100.0, seed=1):
    """
    Fake stars placed uniformly in a sphere, with *known* M_G. We build the
    observables (G, parallax) from those truths, so prepare() should recover
    the true M_G exactly. This checks the code, not astrophysics.
    """
    rng = np.random.default_rng(seed)
    n_ms, n_gi, n_wd = int(n * 0.9), int(n * 0.03), int(n * 0.07)

    # main sequence: rough M_G vs colour relation plus scatter
    c_ms = rng.uniform(0.3, 3.8, n_ms)
    m_ms = 1.0 + 3.6 * c_ms - 0.15 * c_ms ** 2 + rng.normal(0, 0.25, n_ms)
    # red giants and clump
    c_gi = rng.uniform(1.0, 2.4, n_gi)
    m_gi = 0.5 - 2.0 * (c_gi - 1.2) + rng.normal(0, 0.3, n_gi)
    # white dwarfs
    c_wd = rng.uniform(-0.3, 0.9, n_wd)
    m_wd = 11.0 + 3.2 * c_wd + rng.normal(0, 0.3, n_wd)

    color = np.concatenate([c_ms, c_gi, c_wd])
    abs_g = np.concatenate([m_ms, m_gi, m_wd])
    dist = max_pc * rng.uniform(0, 1, color.size) ** (1 / 3)   # uniform in volume
    dist = np.clip(dist, 5.0, None)

    parallax = 1000.0 / dist
    g = abs_g + 5.0 * np.log10(dist / 10.0)
    nu = 120
    return pd.DataFrame({
        "source_id": np.arange(color.size),
        "parallax": parallax,
        "parallax_error": parallax * 0.02,
        "phot_g_mean_mag": g,
        "bp_rp": color,
        "astrometric_chi2_al": np.full(color.size, float(nu - 5)),   # unit weight error = 1
        "astrometric_n_good_obs_al": np.full(color.size, nu),
        "true_abs_g": abs_g,
    })


def test_prepare_recovers_true_absolute_magnitude():
    cat = make_synthetic_catalogue()
    out = hr.prepare(cat)
    assert len(out) == len(cat)
    assert np.allclose(out["abs_g"], out["true_abs_g"], atol=1e-9)
    assert np.allclose(out["distance_pc"], 1000.0 / out["parallax"])
    assert np.allclose(out["abs_g_err"], hr.ABS_MAG_ERR_FACTOR * 0.02)


def test_prepare_drops_bad_parallax_and_nans():
    cat = make_synthetic_catalogue(n=1000)
    cat.loc[0, "parallax"] = -1.0
    cat.loc[1, "bp_rp"] = np.nan
    assert len(hr.prepare(cat)) == len(cat) - 2


def test_feature_anchor_lands_on_the_feature():
    out = hr.prepare(make_synthetic_catalogue())
    # White dwarfs in the fake catalogue: M_G = 11 + 3.2 * colour, so ~11.3 at colour 0.1
    x, y = hr.feature_anchor(out, ((-0.2, 0.5), (9.5, 16.0)), fallback=(99, 99))
    assert -0.2 <= x <= 0.5 and abs(y - (11.0 + 3.2 * x)) < 0.6


def test_feature_anchor_falls_back_on_sparse_data():
    out = hr.prepare(make_synthetic_catalogue(n=1000)).head(10)
    assert hr.feature_anchor(out, ((1.9, 2.1), (6.0, 13.0)), fallback=(2.0, 9.3)) == (2.0, 9.3)


def test_plot_writes_a_real_image():
    out = hr.prepare(make_synthetic_catalogue())
    with tempfile.TemporaryDirectory() as tmp:
        for annotate in (True, False):
            path = hr.plot_hr(out, Path(tmp) / f"hrd_{annotate}.png", 100.0, annotate=annotate,
                              source_label="ESA Gaia DR3 (via a mirror)")
            assert path.exists() and path.stat().st_size > 20_000


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
