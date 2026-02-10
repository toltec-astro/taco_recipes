from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import netCDF4 as nc
import numpy as np
from scipy.signal import welch

from tollan.utils.log import logger, reset_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tolteca_datamodels.toltec.types import ToltecRoachInterface, ToltecArray

if TYPE_CHECKING:
    pass


# The maximum duration of timestream to use for PSD computation (seconds).
PSD_DURATION_SEC = 10.0
# Welch PSD segment length (samples). Smaller = smoother but less resolution.
PSD_NPERSEG = 256


def _find_timestream_files(toltec_fs, obs_spec_tbl):
    """Find timestream (no suffix) netCDF files for each network.

    Returns dict of {roach_index: filepath}.
    """
    files = {}
    for _, row in obs_spec_tbl.iterrows():
        suffix = row.get("file_suffix", "")
        if suffix not in ("", "timestream"):
            continue
        nw = row["roach"]
        fp = Path(row["source"])
        if fp.exists():
            files[nw] = fp
            logger.debug(f"found timestream file for nw {nw}: {fp}")
    return files


@timeit
def _compute_psd_from_file(filepath, duration_sec=PSD_DURATION_SEC, nperseg=PSD_NPERSEG):
    """Read IQ data from a netCDF file and compute PSD waterfall.

    Parameters
    ----------
    filepath : Path
        Path to the toltec netCDF timestream file.
    duration_sec : float
        Maximum duration of data to use (seconds from start).
    nperseg : int
        Welch window segment length.

    Returns
    -------
    dict with keys:
        freqs : (n_freq,) array of PSD frequencies in Hz
        psd : (n_freq, n_chan) PSD waterfall, channels sorted by tone frequency
        tone_freqs_sorted : (n_chan,) sorted tone frequencies in Hz
        sort_idx : (n_chan,) channel sort indices
        fsmp : float, sample rate in Hz
        roach : int, roach index
        lo_freq : float, LO center frequency in Hz
        n_samples_used : int, number of time samples used
    """
    with nc.Dataset(str(filepath), "r") as ds:
        fsmp = float(ds.variables["Header.Toltec.SampleFreq"][:])
        roach = int(ds.variables["Header.Toltec.RoachIndex"][:])
        lo_freq = float(ds.variables["Header.Toltec.LoCenterFreq"][:])
        tone_freqs = ds.variables["Header.Toltec.ToneFreq"][0, :]  # (n_chan,)

        n_time = ds.dimensions["time"].size
        n_samples = min(int(duration_sec * fsmp), n_time)
        logger.debug(
            f"nw {roach}: fsmp={fsmp:.2f} Hz, "
            f"n_time={n_time}, using n_samples={n_samples} "
            f"({n_samples / fsmp:.1f}s), n_chan={len(tone_freqs)}"
        )

        Is = ds.variables["Data.Toltec.Is"][:n_samples, :].astype(np.float64)
        Qs = ds.variables["Data.Toltec.Qs"][:n_samples, :].astype(np.float64)

    # Amplitude of the complex IQ timestream
    amp = np.sqrt(Is**2 + Qs**2)

    # Sort channels by tone frequency
    sort_idx = np.argsort(tone_freqs)
    amp_sorted = amp[:, sort_idx]
    tone_freqs_sorted = tone_freqs[sort_idx]

    # Compute PSD for each channel
    nperseg_use = min(nperseg, n_samples)
    n_chan = amp_sorted.shape[1]
    freqs, psd_0 = welch(amp_sorted[:, 0], fs=fsmp, nperseg=nperseg_use)
    psd = np.zeros((len(freqs), n_chan))
    psd[:, 0] = psd_0
    for ch in range(1, n_chan):
        _, psd[:, ch] = welch(amp_sorted[:, ch], fs=fsmp, nperseg=nperseg_use)

    return {
        "freqs": freqs,
        "psd": psd,
        "tone_freqs_sorted": tone_freqs_sorted,
        "sort_idx": sort_idx,
        "fsmp": fsmp,
        "roach": roach,
        "lo_freq": lo_freq,
        "n_samples_used": n_samples,
    }


@timeit
def _make_noise_figure():
    """Create a 2x7 grid figure for per-network PSD waterfall plots.

    Layout matches reduce_ql.py: top row = a1100 (7 networks),
    bottom row = a1400 (4) + a2000 (2), with shared axes.
    """
    kids_interfaces = ToltecRoachInterface.interfaces  # 13 interfaces

    figsize = (18, 3)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 7, figure=fig, wspace=0.05, hspace=0.35)

    axes = {}
    first_ax = None
    for i, interface in enumerate(kids_interfaces):
        array_name = ToltecArray.interface_array_name[interface]
        if array_name == "a1100":
            row, col = 0, i
        else:
            row = 1
            col = i - 7

        share_kw = {}
        if first_ax is not None:
            share_kw["sharey"] = first_ax

        ax = fig.add_subplot(gs[row, col], **share_kw)
        if first_ax is None:
            first_ax = ax

        is_label_ax = (row == 1 and col == 0)

        # Configure tick labels — each panel shows its own x-axis
        ax.tick_params(
            axis="y", which="both",
            left=True, right=False,
            labelleft=(col == 0),
        )

        axes[interface] = {
            "ax": ax,
            "is_label_ax": is_label_ax,
        }

    return {
        "fig": fig,
        "axes": axes,
        "interfaces": kids_interfaces,
    }


def _plot_psd_waterfall(ax_info, psd_data):
    """Plot a PSD waterfall for one network on the given axes.

    x-axis: channel index (sorted by tone frequency)
    y-axis: PSD frequency (Hz)
    color: log10(PSD)
    """
    ax = ax_info["ax"]
    freqs = psd_data["freqs"]
    psd = psd_data["psd"]
    n_chan = psd.shape[1]

    # Mask zeros for log
    psd_plot = psd.copy()
    psd_plot[psd_plot <= 0] = np.nan

    im = ax.pcolormesh(
        np.arange(n_chan),
        freqs,
        np.log10(psd_plot),
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )

    ax.set_xlim(0, n_chan)
    ax.set_ylim(freqs[1], freqs[-1])  # skip DC bin

    return im


@timeit
def make_noise_quicklook(tbl, output_dir, show_plot=False, **kwargs):
    """Create the noise PSD quicklook figure.

    Parameters
    ----------
    tbl : DataFrame
        Raw obs info table from LmtToltecPathOption.
    output_dir : Path
        Directory for output files.
    show_plot : bool
        If True, show interactively instead of saving.
    """
    # Filter to timestream files only (file_suffix is None/NaN for raw timestreams)
    tbl_ts = tbl[tbl["file_suffix"].isna() | tbl["file_suffix"].isin(["", "timestream"])]
    if len(tbl_ts) == 0:
        logger.warning("no timestream files found, skipping noise QL")
        return None

    obsnum = tbl_ts.iloc[0]["obsnum"]
    ut = tbl_ts.iloc[0]["file_timestamp"]
    logger.info(f"computing noise PSD quicklook for obsnum={obsnum}")

    # Compute PSD for each network
    psd_data = {}
    for _, row in tbl_ts.iterrows():
        nw = row["roach"]
        filepath = Path(row["filepath"])
        if not filepath.exists():
            logger.warning(f"file not found for nw {nw}: {filepath}")
            continue
        try:
            psd_data[nw] = _compute_psd_from_file(filepath)
        except Exception as e:
            logger.error(f"failed PSD for nw {nw}: {e}")

    if not psd_data:
        logger.error("no PSD data computed, abort")
        return None

    # Determine common color scale across all networks
    all_psd = np.concatenate([d["psd"].ravel() for d in psd_data.values()])
    all_psd = all_psd[all_psd > 0]
    vmin = np.log10(np.percentile(all_psd, 1))
    vmax = np.log10(np.percentile(all_psd, 99))
    logger.debug(f"color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # Create figure
    fctx = _make_noise_figure()
    fig = fctx["fig"]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    last_im = None
    for interface in fctx["interfaces"]:
        nw = ToltecRoachInterface.interface_roach[interface]
        array_name = ToltecArray.interface_array_name[interface]
        ax_info = fctx["axes"][interface]
        ax = ax_info["ax"]

        if nw not in psd_data:
            ax.text(
                0.5, 0.5, "NO DATA",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=9, color="gray",
            )
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            d = psd_data[nw]
            freqs = d["freqs"]
            psd = d["psd"]
            n_chan = psd.shape[1]

            psd_plot = psd.copy()
            psd_plot[psd_plot <= 0] = np.nan

            sort_idx = d["sort_idx"]
            im = ax.pcolormesh(
                np.arange(n_chan),
                freqs,
                np.log10(psd_plot),
                shading="auto",
                cmap="viridis",
                norm=norm,
                rasterized=True,
            )
            last_im = im
            ax.set_xlim(0, n_chan)
            ax.set_ylim(freqs[1], freqs[-1])  # skip DC

            # X ticks:
            #   position 0: orig ID of first channel after sorting
            #   sorted position of last orig channel (n_chan-1): label "n_chan-1"
            last_orig_ch = n_chan - 1
            pos_of_last = int(np.where(sort_idx == last_orig_ch)[0][0])
            tick_positions = [0, pos_of_last]
            tick_labels = [str(sort_idx[0]), str(last_orig_ch)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            for lbl, ha in zip(ax.get_xticklabels(), ["left", "center"]):
                lbl.set_ha(ha)

        # Label
        ax.text(
            0.98, 0.95, f"nw{nw}",
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
        )
        if ax_info["is_label_ax"]:
            ax.set_xlabel("chan_id")
            ax.set_ylabel("PSD Freq (Hz)")
            # ax.set_xlabel("Channel ID (sorted by freq)")
            # ax.set_ylabel(r"PSD of $\sqrt{I^2 + Q^2}$ (Hz$^{-1}$)")

    # Colorbar
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.set_label(r"log$_{10}$(PSD of $\sqrt{I^2+Q^2}$)", fontsize=9)

    # fig.suptitle(
        # f"Noise PSD Waterfall  ObsNum={obsnum} ({ut})\n"
        # f"First {PSD_DURATION_SEC:.0f}s, nperseg={PSD_NPERSEG}",
        # fontsize=12,
    # )
    # fig.subplots_adjust(right=0.90)
    fig.tight_layout()

    if show_plot:
        plt.show()
    else:
        outname = f"ql_toltec_{obsnum}_noise_psd.png"
        outfile = Path(output_dir).joinpath(outname)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        logger.info(f"figure saved: {outfile}")

    plt.close(fig)
    return {"output_file": outfile if not show_plot else None}


if __name__ == "__main__":
    import argparse
    from toltec_file_utils import LmtToltecPathOption

    parser = argparse.ArgumentParser(
        description="Generate noise PSD quicklook waterfall plots per network."
    )
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--show_plot",
        action="store_true",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=PSD_DURATION_SEC,
        help="Duration of timestream to use for PSD (seconds).",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=PSD_NPERSEG,
        help="Welch PSD segment length (samples).",
    )
    LmtToltecPathOption.add_obs_spec_argument(parser, required=True, multi=True)
    LmtToltecPathOption.add_data_lmt_path_argument(parser)

    option = parser.parse_args()
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")

    # Override globals from CLI
    PSD_DURATION_SEC = option.duration
    PSD_NPERSEG = option.nperseg

    path_option = LmtToltecPathOption(option)

    tbl = path_option.get_raw_obs_info_table().sort_values("roach")
    if tbl is None:
        logger.error("no valid files specified, exit.")
        sys.exit(1)

    # Filter to timestream files (file_suffix is None/NaN for raw timestreams)
    tbl_ts = tbl[tbl["file_suffix"].isna() | tbl["file_suffix"].isin(["", "timestream"])]
    if len(tbl_ts) == 0:
        logger.error("no timestream files found in obs spec, exit.")
        sys.exit(1)
    logger.info(f"found {len(tbl_ts)} timestream files across networks")
    logger.debug(f"timestream files:\n{tbl_ts}")

    # Create output dir
    uid_obs = tbl_ts.iloc[0]["uid_obs"]
    output_dir = Path(option.output_dir)
    ql_output_dir = output_dir.joinpath(uid_obs)
    if not ql_output_dir.exists():
        ql_output_dir.mkdir(parents=True)

    make_noise_quicklook(
        tbl_ts,
        show_plot=option.show_plot,
        output_dir=ql_output_dir,
    )
