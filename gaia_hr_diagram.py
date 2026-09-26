#!/usr/bin/env python3
"""
Hertzsprung-Russell diagram from real Gaia DR3 data.

Pipeline
--------
1. Query the ESA Gaia archive (gaiadr3.gaia_source) for stars with precise
   parallaxes and clean photometry, out to a chosen distance.
2. Cache the download as a CSV so the archive is only queried once.
3. Apply the last quality cut (astrometric noise) locally.
4. Turn parallax into distance, then apparent G magnitude into absolute
   magnitude M_G using the distance modulus.
5. Plot M_G against BP-RP colour as a log-density map, with the y-axis
   inverted (brighter stars at the top, as in every HR diagram).

The physics in one line:   M_G = G + 5 * log10(parallax_in_mas) - 10

Quality cuts follow Gaia Collaboration, Babusiaux et al. (2018), A&A 616, A10,
Section 2.1 (the Gaia DR2 observational HR diagrams paper), applied here to DR3.

Usage
-----
    python gaia_hr_diagram.py                      # stars within 100 pc
    python gaia_hr_diagram.py --max-distance 200   # larger, slower sample
    python gaia_hr_diagram.py --refresh            # ignore the cached CSV
    python gaia_hr_diagram.py --no-annotations     # plot with no labels
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TABLE = "gaiadr3.gaia_source"
DEFAULT_MAX_DISTANCE_PC = 100.0

# Quality cuts, Babusiaux et al. (2018), Sec. 2.1.
MIN_PARALLAX_OVER_ERROR = 10      # parallax_over_error > 10
MIN_VISIBILITY_PERIODS = 8        # visibility_periods_used > 8
MIN_G_FLUX_SNR = 50               # phot_g_mean_flux_over_error > 50   (sigma_G < 0.022 mag)
MIN_BP_RP_FLUX_SNR = 20           # phot_bp/rp_mean_flux_over_error > 20

# How we wait for the archive's asynchronous job.
JOB_POLL_SECONDS = 10
JOB_TIMEOUT_MINUTES = 60       # a full-catalogue search can take a while when the archive is busy
JOB_TIMEOUT_SECONDS = JOB_TIMEOUT_MINUTES * 60
HTTP_TIMEOUT_SECONDS = 60      # give up on any single silent request after this long

# Backup route: the Heidelberg (ARI) mirror of Gaia DR3, same table and columns.
ARI_TAP_URL = "https://gaia.ari.uni-heidelberg.de/tap"
FALLBACK_MAXREC = 5_000_000    # row cap to request if the service does not state its own

# Propagating parallax error into absolute magnitude:
#   sigma_M = (5 / ln 10) * sigma_parallax / parallax  ~= 2.17 * (sigma_plx / plx)
ABS_MAG_ERR_FACTOR = 5.0 / np.log(10.0)

# Columns we need from the archive.
COLUMNS = [
    "source_id",
    "parallax",
    "parallax_error",
    "phot_g_mean_mag",
    "bp_rp",
    "astrometric_chi2_al",
    "astrometric_n_good_obs_al",
]

# --------------------------------------------------------------------------
# Plot styling
# --------------------------------------------------------------------------
INK = "#1f2328"        # primary text
MUTED = "#59636e"      # secondary text
GRID = "#d8dee4"       # recessive grid
ACCENT = "#c2410c"     # the one highlight colour (the Sun)

XLIM = (-0.6, 4.2)     # BP - RP colour range shown
YLIM = (16.0, -4.0)    # M_G range shown (inverted: bright at the top)

# Approximate sun position in Gaia bands (Casagrande & VandenBerg 2018).
SUN_BP_RP = 0.82
SUN_M_G = 4.67

# Feature labels. Coordinates are (BP-RP, M_G).
#   text        what to print
#   fallback    where the arrow points if the data window below is nearly empty
#   text_pos    where the text sits (chosen to be empty space on the published
#               Gaia DR2 HR diagram)
#   window      ((x0, x1), (y0, y1)): the arrow is aimed at the densest spot
#               inside this box, so it lands on the real feature in your data
# If a label ends up somewhere silly on your plot, nudge text_pos or window here.
LABELS = [
    ("Main sequence",    (2.00, 9.30),  (2.95, 7.00),  ((1.90, 2.10), (6.0, 13.0))),
    ("Red giant branch", (1.10, 2.80),  (2.30, 2.90),  ((1.05, 1.30), (1.6, 3.6))),
    ("Red clump",        (1.20, 0.50),  (0.15, -2.30), ((1.05, 1.35), (-0.5, 1.6))),
    ("White dwarfs",     (0.10, 12.00), (0.55, 8.60),  ((-0.2, 0.5), (9.5, 16.0))),
]
MIN_STARS_FOR_ANCHOR = 30


# --------------------------------------------------------------------------
# Getting the data
# --------------------------------------------------------------------------
def build_query(max_distance_pc: float) -> str:
    """ADQL query: nearby stars with precise astrometry and clean photometry."""
    min_parallax_mas = 1000.0 / max_distance_pc
    return f"""
SELECT {", ".join(COLUMNS)}
FROM {TABLE}
WHERE parallax > {min_parallax_mas:.4f}
  AND parallax_over_error > {MIN_PARALLAX_OVER_ERROR}
  AND visibility_periods_used > {MIN_VISIBILITY_PERIODS}
  AND phot_g_mean_flux_over_error > {MIN_G_FLUX_SNR}
  AND phot_bp_mean_flux_over_error > {MIN_BP_RP_FLUX_SNR}
  AND phot_rp_mean_flux_over_error > {MIN_BP_RP_FLUX_SNR}
  AND phot_bp_rp_excess_factor > 1.0 + 0.015 * POWER(bp_rp, 2)
  AND phot_bp_rp_excess_factor < 1.3 + 0.06 * POWER(bp_rp, 2)
""".strip()


def wait_for_job(job, timeout_s: float = JOB_TIMEOUT_SECONDS, poll_s: float = JOB_POLL_SECONDS,
                 sleep=None, clock=None, say=print) -> None:
    """
    Poll an asynchronous Gaia archive job until it finishes, printing its status.

    Returns when the job is COMPLETED. Raises RuntimeError if the archive ends
    the job with ERROR/ABORTED or stops answering, and TimeoutError if it takes
    longer than timeout_s. (sleep/clock/say are parameters so this can be tested
    without waiting.)
    """
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    start = clock()
    last_phase, last_print, failures = None, None, 0
    while True:
        try:
            phase = str(job.get_phase(update=True)).strip().upper()
            failures = 0
        except Exception as exc:  # a brief network blip should not kill a long wait
            failures += 1
            if failures >= 5:
                raise RuntimeError(f"lost contact with the archive while waiting ({exc})") from exc
            say(f"  no answer from the archive ({type(exc).__name__}), retrying [{failures}/5]")
            sleep(poll_s)
            continue

        elapsed = clock() - start
        if phase != last_phase or last_print is None or elapsed - last_print >= 30:
            say(f"  [{elapsed:5.0f}s] archive job status: {phase}")
            last_phase, last_print = phase, elapsed

        if phase == "COMPLETED":
            return
        if phase in ("ERROR", "ABORTED"):
            raise RuntimeError(f"the archive ended the job with status {phase}")
        if elapsed >= timeout_s:
            raise TimeoutError(f"gave up after {timeout_s / 60:.0f} minutes (last status: {phase})")
        sleep(poll_s)


def fetch_from_gaia(max_distance_pc: float,
                    timeout_minutes: float = JOB_TIMEOUT_MINUTES) -> pd.DataFrame:
    """Run the query as an asynchronous job on the Gaia archive."""
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        sys.exit("astroquery is not installed. Run: pip install -r requirements.txt")

    query = build_query(max_distance_pc)
    print(f"Submitting query to the ESA Gaia archive (stars within {max_distance_pc:g} pc)...")
    print("This is a search of the whole catalogue, so it can take from a minute to 30+ minutes")
    print(f"when the archive is busy (this script waits up to {timeout_minutes:g} minutes).")
    print("Keep the Mac awake and this window open. A status line is printed below as it runs.")
    print(query, end="\n\n")

    # Async jobs are not limited to 2000 rows like synchronous ones.
    # (Anonymous users are capped at about 3 million rows per job.)
    # background=True returns as soon as the job is created, so we can report
    # its status ourselves instead of sitting silent until it finishes.
    #
    # astroquery opens its connections with no timeout, so a silent server makes
    # it wait forever. A default socket timeout turns that into an error that
    # wait_for_job() can retry.
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(HTTP_TIMEOUT_SECONDS)
    try:
        job = Gaia.launch_job_async(query, background=True)
        print(f"Archive job id: {getattr(job, 'jobid', 'unknown')}")
        wait_for_job(job, timeout_s=timeout_minutes * 60)
        table = job.get_results()
    except Exception as exc:  # network down, archive overloaded, query rejected...
        sys.exit(
            f"\nCould not get data from the Gaia archive: {type(exc).__name__}: {exc}\n"
            "The archive is sometimes overloaded, so trying again in a few minutes often works. "
            "Also check your internet connection."
        )
    finally:
        socket.setdefaulttimeout(previous_timeout)
    df = table.to_pandas()
    print(f"Downloaded {len(df):,} rows.")
    return df


class _PyvoJob:
    """Wraps a pyvo job so wait_for_job() can treat it like an astroquery job."""

    def __init__(self, job):
        self.job = job
        self.jobid = getattr(job, "job_id", "unknown")

    def get_phase(self, update=False):
        return self.job.phase          # pyvo re-asks the server every time


def fetch_from_ari(max_distance_pc: float,
                   timeout_minutes: float = JOB_TIMEOUT_MINUTES) -> pd.DataFrame:
    """
    Same query, run on the Heidelberg (ARI) mirror of Gaia DR3 instead of ESA's
    archive. Useful when ESA's archive is slow. Uses pyvo, the standard Python
    client for TAP services.
    """
    try:
        import pyvo
    except ImportError:
        sys.exit("pyvo is not installed. Run: pip install -r requirements.txt")

    query = build_query(max_distance_pc)
    print(f"Submitting query to the Heidelberg (ARI) mirror of Gaia DR3 (stars within {max_distance_pc:g} pc)...")
    print(f"This script waits up to {timeout_minutes:g} minutes. Keep the Mac awake and this window open.")
    print(query, end="\n\n")

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(HTTP_TIMEOUT_SECONDS)
    try:
        service = pyvo.dal.TAPService(ARI_TAP_URL)
        try:
            limit = int(service.hardlimit)       # the most rows the service will return
        except Exception:
            limit = FALLBACK_MAXREC
        job = service.submit_job(query, maxrec=limit)
        job.run()
        print(f"Job id: {job.job_id}")
        wait_for_job(_PyvoJob(job), timeout_s=timeout_minutes * 60)
        job.raise_if_error()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = job.fetch_result().to_table()
        overflowed = any(issubclass(w.category, pyvo.dal.DALOverflowWarning) for w in caught)
    except Exception as exc:
        sys.exit(
            f"\nCould not get data from the ARI mirror: {type(exc).__name__}: {exc}\n"
            "Trying again in a few minutes often works. Also check your internet connection."
        )
    finally:
        socket.setdefaulttimeout(previous_timeout)

    if overflowed or len(table) >= limit:
        sys.exit(
            f"\nThe service cut the result off at {limit:,} rows, so the sample would be incomplete "
            "and the plot misleading. Try a smaller --max-distance (for example 50)."
        )
    df = table.to_pandas()
    print(f"Downloaded {len(df):,} rows.")
    return df


def load_data(max_distance_pc: float, data_path: str | None, refresh: bool,
              timeout_minutes: float = JOB_TIMEOUT_MINUTES, source: str = "esa") -> pd.DataFrame:
    """Use an explicit CSV, else a cached download, else query the chosen archive."""
    if data_path:
        path = Path(data_path)
        if not path.exists():
            sys.exit(f"--data file not found: {path}")
        print(f"Reading {path}")
        return pd.read_csv(path)

    cache = Path("data") / f"gaia_dr3_within_{max_distance_pc:g}pc_{source}.csv"
    if cache.exists() and not refresh:
        print(f"Using cached download: {cache}  (use --refresh to re-query)")
        return pd.read_csv(cache)

    fetch = fetch_from_ari if source == "ari" else fetch_from_gaia
    df = fetch(max_distance_pc, timeout_minutes)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"Saved a copy to {cache}")
    return df


# --------------------------------------------------------------------------
# The physics
# --------------------------------------------------------------------------
def absolute_magnitude(g_mag, parallax_mas):
    """
    Absolute G magnitude from apparent magnitude and parallax (in milliarcseconds).

    Distance modulus:  m - M = 5 * log10(d / 10 pc)
    Distance:          d = 1000 / parallax_mas   (parsecs)
    Combining:         M = m + 5 * log10(parallax_mas) - 10
    """
    return np.asarray(g_mag) + 5.0 * np.log10(np.asarray(parallax_mas)) - 10.0


def astrometric_cut(df: pd.DataFrame) -> pd.Series:
    """
    Reject sources with a poor astrometric fit (Babusiaux et al. 2018):

        sqrt(chi2 / (nu - 5)) < 1.2 * max(1, exp(-0.2 * (G - 19.5)))

    where chi2 = astrometric_chi2_al and nu = astrometric_n_good_obs_al.
    Done in pandas rather than ADQL so it is easy to read and to test.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_weight_error = np.sqrt(
            df["astrometric_chi2_al"] / (df["astrometric_n_good_obs_al"] - 5.0)
        )
        limit = 1.2 * np.maximum(1.0, np.exp(-0.2 * (df["phot_g_mean_mag"] - 19.5)))
        keep = unit_weight_error < limit
    return keep.fillna(False)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the local cut, then add distance and absolute magnitude columns."""
    needed = ["parallax", "parallax_error", "phot_g_mean_mag", "bp_rp"]
    out = df.dropna(subset=needed).copy()
    out = out[out["parallax"] > 0]
    out = out[astrometric_cut(out)].copy()

    out["distance_pc"] = 1000.0 / out["parallax"]
    out["abs_g"] = absolute_magnitude(out["phot_g_mean_mag"], out["parallax"])
    out["abs_g_err"] = ABS_MAG_ERR_FACTOR * out["parallax_error"] / out["parallax"]
    return out


def print_summary(n_downloaded: int, df: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Rows in file / download:      {n_downloaded:,}")
    print(f"Stars plotted after cuts:     {len(df):,}")
    print(f"Distance range:               {df['distance_pc'].min():.1f} - {df['distance_pc'].max():.1f} pc "
          f"(median {df['distance_pc'].median():.1f} pc)")
    print(f"M_G range:                    {df['abs_g'].min():.2f} to {df['abs_g'].max():.2f} mag")
    print(f"BP-RP range:                  {df['bp_rp'].min():.2f} to {df['bp_rp'].max():.2f} mag")
    print(f"Median M_G uncertainty:       {df['abs_g_err'].median():.3f} mag "
          f"(worst case {df['abs_g_err'].max():.3f})")


# --------------------------------------------------------------------------
# The plot
# --------------------------------------------------------------------------
def _density_cmap() -> ListedColormap:
    """One hue, light to dark. Start above white so single stars stay visible."""
    blues = matplotlib.colormaps["Blues"]
    return ListedColormap(blues(np.linspace(0.30, 1.0, 256)))


def feature_anchor(df: pd.DataFrame, window, fallback):
    """
    Centre of the densest spot of stars inside window = ((x0, x1), (y0, y1)),
    in (BP-RP, M_G). Falls back to fixed coordinates if the window is too empty.
    """
    (x0, x1), (y0, y1) = window
    box = df[(df["bp_rp"] >= x0) & (df["bp_rp"] <= x1) &
             (df["abs_g"] >= y0) & (df["abs_g"] <= y1)]
    if len(box) < MIN_STARS_FOR_ANCHOR:
        return fallback
    h, xe, ye = np.histogram2d(box["bp_rp"], box["abs_g"], bins=(4, 20),
                               range=((x0, x1), (y0, y1)))
    i, j = np.unravel_index(np.argmax(h), h.shape)
    return (0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1]))


def plot_hr(df: pd.DataFrame, out_path: Path, max_distance_pc: float,
            annotate: bool = True, source_label: str = "ESA Gaia DR3") -> Path:
    """Draw the HR diagram as a log-density map and save it as a PNG."""
    fig, ax = plt.subplots(figsize=(8.0, 9.6), dpi=150)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6)

    # Density map: count stars in small colour/magnitude bins, log colour scale.
    counts = ax.hist2d(
        df["bp_rp"], df["abs_g"],
        bins=(220, 300),
        range=((XLIM[0], XLIM[1]), (YLIM[1], YLIM[0])),
        norm=LogNorm(), cmap=_density_cmap(), cmin=1,
    )
    # Colourbar the same height as the plot.
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
    cbar = fig.colorbar(counts[3], cax=cax)
    cbar.set_label("Stars per bin (log scale)", color=MUTED, fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)  # YLIM is (16, -4), so this inverts the axis
    ax.set_xlabel("BP − RP colour (mag)      bluer, hotter ←   → redder, cooler",
                  color=INK, fontsize=11)
    # Rotated text reads upwards, so a right-pointing arrow ends up pointing up.
    ax.set_ylabel("Absolute G magnitude, $M_G$ (mag)      brighter →", color=INK, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.set_title("Hertzsprung–Russell diagram from Gaia DR3",
                 loc="left", fontsize=15, fontweight="bold", color=INK, pad=30)
    ax.text(0.0, 1.015,
            f"{len(df):,} stars within {max_distance_pc:g} pc  ·  "
            f"median $M_G$ uncertainty {df['abs_g_err'].median():.3f} mag",
            transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")

    if annotate:
        for text, fallback, text_pos, window in LABELS:
            target = feature_anchor(df, window, fallback)
            ax.annotate(
                text, xy=target, xytext=text_pos,
                fontsize=10.5, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=4, shrinkB=2),
                zorder=5,
            )
        ax.plot(SUN_BP_RP, SUN_M_G, marker="o", markersize=9, markerfacecolor="none",
                markeredgecolor=ACCENT, markeredgewidth=1.8, zorder=6)
        ax.annotate("Sun", xy=(SUN_BP_RP, SUN_M_G), xytext=(SUN_BP_RP + 0.35, SUN_M_G - 0.9),
                    fontsize=10.5, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9, shrinkA=2, shrinkB=6),
                    zorder=6)

    fig.text(0.01, 0.006,
             f"Data: {source_label}. Quality cuts after Gaia Collaboration, Babusiaux et al. (2018).",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    return out_path


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make an HR diagram from real Gaia DR3 data.")
    p.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE_PC, metavar="PC",
                   help="only stars closer than this many parsecs (default: %(default)g). "
                        "100-200 is quick; larger samples are much bigger downloads.")
    p.add_argument("--data", metavar="CSV",
                   help="use this CSV instead of querying the archive (same columns as the cache).")
    p.add_argument("--refresh", action="store_true",
                   help="ignore any cached download and query the archive again.")
    p.add_argument("--source", choices=("esa", "ari"), default="esa",
                   help="where to get the data: 'esa' (the official Gaia archive, default) or 'ari' "
                        "(a Heidelberg University mirror of Gaia DR3, handy if ESA is slow).")
    p.add_argument("--timeout", type=float, default=JOB_TIMEOUT_MINUTES, metavar="MIN",
                   help="give up waiting for the archive after this many minutes (default: %(default)g).")
    p.add_argument("--out", default="figures/hr_diagram.png", metavar="PNG",
                   help="where to save the figure (default: %(default)s)")
    p.add_argument("--no-annotations", action="store_true",
                   help="leave off the feature labels and the Sun marker.")
    p.add_argument("--show", action="store_true", help="also open the figure in a window.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.max_distance <= 0:
        sys.exit("--max-distance must be positive.")

    raw = load_data(args.max_distance, args.data, args.refresh, args.timeout, args.source)
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        sys.exit(f"Data is missing columns: {missing}")

    df = prepare(raw)
    if df.empty:
        sys.exit("No stars left after the quality cuts.")
    print_summary(len(raw), df)

    source_label = ("ESA Gaia DR3 (via the Heidelberg ARI mirror)" if args.source == "ari" and not args.data
                    else "ESA Gaia DR3")
    out = plot_hr(df, Path(args.out), args.max_distance, annotate=not args.no_annotations,
                  source_label=source_label)
    print(f"\nSaved figure to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
