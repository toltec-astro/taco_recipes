from __future__ import annotations
from astropy.table import QTable
import sys
import astropy.units as u
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from tollan.utils.log import logger, reset_logger, timeit
from tollan.utils.fmt import pformat_yaml
from pathlib import Path
from tolteca_datamodels.toltec.types import ToltecRoachInterface, ToltecArray
from tolteca_kids.kids_find import SegmentBitMask
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("agg")
import matplotlib.transforms as mtrans
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

if TYPE_CHECKING:
    from tolteca_datamodels.toltec.file import SourceInfoDataFrame, SourceInfoModel


def _find_file(patterns, search_paths, subpaths=None, unique=True):
    logger.debug(f"search patterns in paths:\n{pformat_yaml(patterns)}")
    files = []
    subpaths = subpaths or [""]
    for path in search_paths:
        for sp in subpaths:
            pp = Path(path).joinpath(sp)
            for pattern in patterns:
                ff = list(pp.glob(pattern))
                if ff:
                    logger.debug(f"found files in {pp}:\n{pformat_yaml(ff)}")
                files.extend(ff)
                if unique:
                    if len(files) == 1:
                        return files[0]
    if files:
        return files
    return None


def _ensure_find_file(*args, timeout=60, **kwargs):
    wait = 10
    counter = 0
    import time

    while True:
        file = _find_file(*args, **kwargs)
        if file is None:
            if counter * wait > timeout:
                logger.debug(f"waiting for file {args} timeout")
                return None
            logger.debug(f"waiting for file {args} ...")
            time.sleep(wait)
            counter += 1
            continue
        return file


@timeit
def _collect_kids_info(entry: SourceInfoModel, search_paths, timeout=60):
    interface = entry.interface
    nw = entry.roach
    obsnum = entry.obsnum
    subobsnum = entry.subobsnum
    scannum = entry.scannum

    logger.debug(f"collect kids info for {interface}_{obsnum}_{subobsnum}_{scannum}")
    logger.debug(f"search paths:\n{pformat_yaml(search_paths)}")
    prefix = f"{interface}_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}_"
    subpaths = ["", entry.uid_raw_obs, entry.uid_obs]

    def _load_table(t):
        if t is not None:
            logger.info(f"load table {t}")
            return QTable.read(t, format="ascii")
        logger.warning("missing table")
        return None

    tonelist_table = _load_table(
        _ensure_find_file(
            [f"{prefix}*_kids_find.ecsv"],
            search_paths,
            subpaths=subpaths,
            timeout=timeout,
        ),
    )
    checktone_table = _load_table(
        _find_file(
            [f"{prefix}*_chan_prop.ecsv"],
            search_paths,
            subpaths=subpaths,
        )
    )
    kidscpp_table = _load_table(
        _find_file(
            [
                f"{prefix}*_vnasweep.txt",
                f"{prefix}*_targsweep.txt",
                f"{prefix}*_tune.txt",
            ],
            search_paths,
            subpaths=subpaths,
        )
    )
    # targfreqs_table = _load_table(
    #     _find_file([f"{prefix}*_targfreqs.ecsv"], search_paths)
    # )
    # kidsmodel_table = _load_table(
    #     _find_file(
    #         [
    #             f"{prefix}*_kmt.ecsv",
    #         ],
    #         search_paths,
    #     )
    # )
    # context_data = _find_file(
    #     [f"{prefix}*_ctx.pkl"],
    #     search_paths,
    #     subpaths=subpaths,
    # )
    return locals()


@timeit
def _make_kids_figure(layouts):

    nrows = len(layouts)
    if nrows == 1:
        figsize = (15, 5.5)
        gi_array = gi_nw = 0
    elif nrows == 2:
        figsize = (16, 11)
        gi_array = 0
        gi_nw = 1
    else:
        raise
        # figsize = (16, 10)

    with timeit("make matplotlib figure"):
        fig = plt.figure(figsize=figsize)
        gs0 = gridspec.GridSpec(nrows, 1, figure=fig)

    array_names = ToltecArray.array_names
    kids_interfaces = ToltecRoachInterface.interfaces

    cmap_name = "RdYlGn"
    cmap = matplotlib.colormaps[cmap_name]
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)

    gss = dict()
    axes = dict()
    if "array" in layouts:
        gs = gss["array"] = gs0[gi_array, 0].subgridspec(
            1, len(array_names), wspace=0.01, hspace=0.01
        )

        axes_array = dict()
        kw = {}
        for i, array_name in enumerate(array_names):
            logger.debug(f"make layout for {array_name}")
            dd = {
                # 'is_lim_ax': True,
                "is_label_ax": True,
            }
            if axes_array:
                kw["sharex"] = next(iter(axes_array.values()))["ax"]
                kw["sharey"] = next(iter(axes_array.values()))["ax"]
            else:
                pass
                # dd['is_lim_ax'] = False
            # print(kw)
            ax = fig.add_subplot(gs[0, i], **kw)
            dd.update({"ax": ax, "cmap": cmap})
            # dd.update(toltec_info[array_name])
            if i == 0:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=True,
                    labelbottom=False,
                    labeltop=True,
                )
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=True,
                    right=False,
                    labelleft=True,
                    labelright=False,
                )
            else:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labeltop=False,
                )
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )
                dd["is_label_ax"] = False
            axes_array[array_name] = dd
        axes.update(axes_array)
    if "nw" in layouts:
        gs = gss["nw"] = gs0[gi_nw, 0].subgridspec(2, 7, wspace=0.01, hspace=0.01)
        axes_nw = dict()
        kw = {}
        for i, interface in enumerate(kids_interfaces):
            logger.debug(f"make layout for {interface}")
            dd = {
                # 'is_lim_ax': True,
                "is_label_ax": True,
            }
            if axes_nw:
                kw["sharex"] = next(iter(axes_nw.values()))["ax"]
                kw["sharey"] = next(iter(axes_nw.values()))["ax"]
            else:
                pass
                # dd['is_lim_ax'] = False
            if ToltecArray.interface_array_name[interface] == "a1100":
                ii = 0
            else:
                ii = 1
                i = i - 7
            ax = fig.add_subplot(gs[ii, i], **kw)
            dd.update({"ax": ax, "cmap": cmap})
            # dd.update(toltec_info[interface])
            if ii == 1 and i == 0:
                pass
            elif i == 0:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labeltop=False,
                )
                dd["is_label_ax"] = False
            else:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labeltop=False,
                )
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                )
                dd["is_label_ax"] = False
            axes_nw[interface] = dd
        axes.update(axes_nw)
    return {
        "fig": fig,
        "gss": gss,
        "axes": axes,
        "array_names": array_names,
        "interfaces": kids_interfaces,
        "cmap": cmap,
    }


def _plot_finding_ratio_nw(
    ax_nw, ax_array, data, phi_lim=5 << u.deg, phi_lim2=15 << u.deg
):
    # make a path outline the
    edge_indices = data["edge_indices"]

    tlt = data.get("tonelist_table", None)
    # tft = data.get("targfreqs_table", None)
    apt = data.get("apt", None)
    tct = data.get("checktone_table", None)
    kct = data.get("kidscpp_table", None)
    # logger.debug(f"data:\n{pformat_yaml(data)}")
    if any(
        [
            tlt is None,
            apt is None,
            tct is None,
            # kct is None,
        ]
    ):
        # msg = [tft is None, apt is None, tct is None, kct is None]
        # logger.debug(f"skip plot: [tft, apt, tct, kct]: {msg}")
        msg = [tlt is None, apt is None, tct is None]
        logger.debug(f"skip plot: [tlt, apt, tct]: {msg}")
        return
    n_found = len(tlt)

    if kct is not None:
        Qr = kct["Qr"]
        r_med = np.median(0.5 / kct["Qr"])
    else:
        Qr = tlt["Qr"]
        r_med = np.median(0.5 / tlt["Qr"])

    x_off = (tlt["dist"] / tlt["f_chan"]).to_value(u.dimensionless_unscaled)
    # phi_off = tlt["d_phi"]

    if len(tct) == 1000:
        # vna sweep
        phi_off = np.full((len(tlt),), 0.0) << u.deg
        m_miss = np.zeros((len(tlt),), dtype=bool)
    else:
        phi_off = (np.arctan2(x_off, r_med) << u.rad).to(u.deg)
        m_miss = np.abs(tlt["dist"]) > (88 << u.kHz)
    n_miss = np.sum(m_miss)
    # print(phi_off)
    m_good = np.abs(phi_off) < phi_lim
    n_good = np.sum(m_good)

    # m_ok = (np.abs(phi_off) < phi_lim2) & (np.abs(phi_off) >= phi_lim)
    # n_ok = np.sum(m_ok) + n_good
    m_ok = np.abs(phi_off) < phi_lim2
    n_ok = np.sum(m_ok)

    m_single = (tlt["bitmask_det"] & SegmentBitMask.blended) == 0
    n_single = np.sum(m_single & (~m_miss))
    m_dup = ~m_single
    n_dup = np.sum(m_dup & (~m_miss))

    n_design = len(apt)
    frac_found = n_found / n_design
    frac_in_tune = n_good / n_design

    phi_lim_v = phi_lim.to_value(u.deg)
    bins = np.arange(-90 - phi_lim_v / 2, 90 + phi_lim_v / 2 + 0.1, phi_lim_v)
    phi_off_v = phi_off.to_value(u.deg)

    # print(phi_off_v[m_single])
    # print(phi_off_v[m_dup])
    # print(phi_off_v[m_miss])
    ax = ax_nw["ax"]
    cmap = ax_nw["cmap"]
    ax.axvspan(-90, -phi_lim2.to_value(u.deg), color="gray", alpha=0.2)
    ax.axvspan(phi_lim2.to_value(u.deg), 90, color="gray", alpha=0.2)

    ax.axvspan(
        -phi_lim2.to_value(u.deg), -phi_lim.to_value(u.deg), color="#cdcdcd", alpha=0.2
    )
    ax.axvspan(
        phi_lim.to_value(u.deg), phi_lim2.to_value(u.deg), color="#cdcdcd", alpha=0.2
    )

    ax.axvline(0.0, linestyle="-", color="black", linewidth=1)
    ax.hist(
        [
            phi_off_v[m_single & m_good],
            phi_off_v[m_dup & m_good],
            phi_off_v[m_single & ~m_good],
            phi_off_v[m_dup & ~m_good],
            # phi_off_v[m_miss],
        ],
        bins=bins,
        color=[cmap(1.0), cmap(0), cmap(0.75), cmap(0.25)],
        stacked=True,
    )
    ax.text(
        0.0,
        1.0,
        f"""
  Design: {n_design:3d}
   Found: {n_found:3d}
 In-tune: {n_good:3d}
 OK-tune: {n_ok:3d}
  Single: {n_single:3d}
Multiple: {n_dup:3d}
  Missed: {n_miss:3d}
""".strip(
            "\n"
        ),
        ha="left",
        va="top",
        fontsize=8,
        fontfamily="monospace",
        transform=ax.transAxes,
    )

    # array outline
    ax = ax_array["ax"]
    cmap = ax_array["cmap"]
    fillcolor = cmap(frac_found)
    ax.fill(
        apt["x_t"][edge_indices].to_value(u.arcsec),
        apt["y_t"][edge_indices].to_value(u.arcsec),
        color=cmap(frac_found),
    )
    return locals()


def _plot_finding_ratio_array(ax_array, nw_ctxs, Qr_lims):
    ax = ax_array["ax"]
    cmap = ax_array["cmap"]
    if not nw_ctxs:
        ax = ax_array["ax"]
        ax.text(
            0.5,
            0.5,
            "NO DATA",
            ha="center",
            va="center",
            transform=ax.transAxes,
            # fontsize=20,
            color=cmap(0),
        )
        return
    n_design = 0
    n_found = 0
    n_good = 0
    n_ok = 0
    for ctx in nw_ctxs:
        n_design += ctx["n_design"]
        n_found += ctx["n_found"]
        n_good += ctx["n_good"]
        n_ok += ctx["n_ok"]
    frac_found = n_found / n_design
    frac_in_tune = n_good / n_design
    frac_ok_tune = n_ok / n_design

    ax.text(
        0.5,
        0.5,
        f"""
Found: {n_found:3d} / {n_design:3d} {frac_found:4.1%}
In-tune: {n_good:3d}/ {n_design:3d} {frac_in_tune:4.1%}
OK-tune: {n_ok:3d}/ {n_design:3d} {frac_ok_tune:4.1%}
""".strip(
            "\n"
        ),
        ha="center",
        va="center",
        # fontsize=20,
        fontfamily="monospace",
        transform=ax.transAxes,
    )

    # add a Qr plot
    ax_divider = make_axes_locatable(ax)
    ax = ax_divider.append_axes("bottom", size="10%", pad="10%")
    Qrs = []
    for ctx in nw_ctxs:
        Qrs.append(ctx["Qr"])
    Qrs = np.hstack(Qrs)
    lim_left, lim_right = Qr_lims
    ax.set_xlim(lim_left - 3000, lim_right + 3000)
    if ax_array["is_label_ax"]:
        ax.set_ylabel("Qr")
        ax.yaxis.label.set(rotation='horizontal', ha='right', va="center")
    ax.set_ylim(-0.5, 0.75)
    # ax.set_xscale("log")
    ax.get_yaxis().set_ticks([])
    # ax.grid(axis='x', which="both")
    ax.axvspan(ax.get_xlim()[0], lim_left, color="#cdcdcd")
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    text_y = 0.95  
    ax.text(lim_left - 1000, text_y, "ROOM", ha="right", va="top", transform=trans)
    ax.axvspan(lim_right, ax.get_xlim()[1], color="#cdcdcd")
    ax.text(lim_right + 1000, text_y, "DARK", ha="left", va="top", transform=trans)
    ax.text((lim_right + lim_left) * 0.5, text_y, "SKY", ha="center", va="top", transform=trans)

    parts = ax.violinplot(
        [Qrs], positions=[0], orientation="horizontal",
        showextrema=False, showmedians=False,
        )
    Qr_med = np.median(Qrs)
    Qr_25, Qr_75 = np.quantile(Qrs, [0.25, 0.75])
    color = cmap(float(Qr_med > lim_left))
    # ax.scatter(np.quantile(Qrs, [0.25, 0.75]), [0, 0], marker='.', color='red')
    ax.errorbar(
        [Qr_med], [0], xerr=[[Qr_med - Qr_25], [Qr_75 - Qr_med]],
        marker='*', color=color, ms=20
        )
    for pc in parts['bodies']:
        pc.set_facecolor(color)



def _make_per_nw_apt_info(nw, apt):
    sel = apt["nw"] == nw
    utbl = apt[sel]
    # polygon region
    # build a polygon to describe the outline
    rows = np.sort(np.unique(utbl["i"]))
    verts = []
    ui = np.where(utbl["ori"] == 0)[0]
    etbl = utbl[ui]
    for row in rows:
        # import pdb
        # pdb.set_trace()
        rv = sorted(np.where(etbl["i"] == row)[0], key=lambda i: etbl[i]["j"])
        if len(verts) > 0:
            verts = rv[:1] + verts + rv[-1:]
        else:
            if len(rv) >= 2:
                verts = [rv[0], rv[-1]]
            else:
                verts = rv
    verts = ui[verts]
    return {
        "select": sel,
        "apt": utbl,
        "edge_indices": verts,
    }


def _make_kids_plot(
    tbl: SourceInfoDataFrame,
    apt_design=None,
    show_plot=True,
    output_dir=None,
    search_paths=None,
):
    # index by nw
    nws = np.unique(tbl["roach"])

    nws_all = ToltecRoachInterface.roaches

    logger.debug(f"make plot for {len(nws)} / {len(nws_all)} kids data.")

    kids_info = {nw: {} for nw in nws}

    search_paths = search_paths or []
    if output_dir is not None:
        search_paths.append(Path(output_dir))
    for source_info in tbl.toltec_file.to_info_list():
        nw = source_info.roach
        search_paths = search_paths + [source_info.filepath.parent]
        kids_info[nw].update(_collect_kids_info(source_info, search_paths))
        if apt_design is not None:
            kids_info[nw].update(_make_per_nw_apt_info(nw, apt_design))

    fctx = _make_kids_figure(layouts=["nw", "array"])

    nw_ctxs_per_array = {array_name: [] for array_name in ToltecArray.array_names}
    for interface in fctx["interfaces"]:
        logger.debug(f"working on {interface=}")
        array_name = ToltecArray.interface_array_name[interface]
        nw = ToltecRoachInterface.interface_roach[interface]
        ax_nw = fctx["axes"][interface]
        ax_array = fctx["axes"][array_name]
        d = kids_info.get(nw, None)
        phi_lim = 5 << u.deg
        if d is None:
            logger.debug(f"no date found for {interface=}")
            ax = ax_nw["ax"]
            cmap = ax_nw["cmap"]
            ax.text(
                0.5,
                0.5,
                "NO DATA",
                ha="center",
                va="center",
                transform=ax.transAxes,
                # fontsize=20,
                color=cmap(0.0),
            )
        else:
            ctx = _plot_finding_ratio_nw(
                ax_array=ax_array, ax_nw=ax_nw, data=d, phi_lim=phi_lim
            )
            if ctx is not None:
                nw_ctxs_per_array[array_name].append(ctx)
            else:
                logger.debug(f"no finding ratio nw plot generated for {interface=}")
        ax = ax_nw["ax"]
        ax.text(1, 1, f"{interface}", ha="right", va="top", transform=ax.transAxes)
        if ax_nw["is_label_ax"]:
            ax.set_xlabel("Phase Offset $atan^{-1}(x / r)$ (deg)")
            ax.set_ylabel("Num. Dets")
            ax.set_xlim(-90, 90)
            ax.set_ylim(0, 350)

    for array_name in ToltecArray.array_names:
        ax_array = fctx["axes"][array_name]
        ax = ax_array["ax"]

        ax.text(1, 1, f"{array_name}", ha="right", va="top", transform=ax.transAxes)
        ax_array["ax"].set_aspect("equal")
        if ax_array["is_label_ax"]:
            ax.set_xlabel("$\Delta lon$ (arcsec)")
            ax.set_ylabel("$\Delta lat$ (arcsec)")
            fov = ToltecArray.fov_diameter
            ax.set_xlim(
                -(fov / 2).to_value(u.arcsec),
                (fov / 2).to_value(u.arcsec),
            )
            ax.set_ylim(
                -(fov / 2).to_value(u.arcsec),
                (fov / 2).to_value(u.arcsec),
            )

    # plot aggregated stats for array
    # TODO: use skydip data result
    Qr_on_sky_lims = {
        "a1100": [8000, 12000],
        "a1400": [4500, 8000],
        "a2000": [4000, 6000],
    }

    for array_name, nw_ctxs in nw_ctxs_per_array.items():
        ax_array = fctx["axes"][array_name]
        _plot_finding_ratio_array(ax_array=ax_array, nw_ctxs=nw_ctxs, Qr_lims=Qr_on_sky_lims[array_name])

    fig = fctx["fig"]
    obsnum = tbl.iloc[0]["obsnum"]
    ut = tbl.iloc[0]["file_timestamp"]
    fig.suptitle(f"KIDs Summary ObsNum={obsnum} ({ut})")
    fig.tight_layout()
    if show_plot:
        plt.show()
    else:
        # save the plot
        outname = f"ql_toltec_{obsnum}_kidsinfo.png"
        outfile = output_dir.joinpath(outname)
        fig.savefig(outfile)
        logger.info(f"figure saved: {outfile}")
    return locals()


def make_quicklook_prod(tbl, **kwargs):
    dp = {}
    dp["kids"] = _make_kids_plot(tbl, **kwargs)
    return dp


def _get_or_create_default_apt(apt_filepath):
    apt_filepath = Path(apt_filepath)
    if apt_filepath.exists():
        logger.debug(f"use default apt {apt_filepath}")
        return QTable.read(apt_filepath, format="ascii.ecsv")
    raise NotImplementedError
    # TODO: re-enable this
    # from tolteca.simu.toltec import ToltecObsSimulatorConfig
    #
    # simulator = ToltecObsSimulatorConfig.from_dict({}).simulator
    # apt = simulator.array_prop_table
    # # cache apt
    # apt.write(apt_filepath, format="ascii.ecsv", overwrite=True)
    # logger.debug(f"create and save default apt to {apt_filepath}")
    # return apt


if __name__ == "__main__":
    import argparse
    from toltec_file_utils import LmtToltecPathOption

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument(
        "--apt_design",
    )
    parser.add_argument(
        "--search_paths",
        nargs="*",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--show_plot",
        action="store_true",
    )
    LmtToltecPathOption.add_obs_spec_argument(parser, required=True, multi=True)
    LmtToltecPathOption.add_data_lmt_path_argument(parser)

    option = parser.parse_args()
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")
    path_option = LmtToltecPathOption(option)

    tbl = path_option.get_raw_obs_info_table().sort_values("roach")
    if tbl is None:
        logger.error("no valid files specified, exit.")
        sys.exit(1)
    tbl = tbl.query("file_suffix in ['tune', 'targsweep', 'vnasweep']")
    if len(np.unique(tbl["uid_raw_obs"])) > 1:
        logger.error("files are not from one single raw obs")
        sys.exit(1)
    logger.debug(f"loaded raw obs files:\n{tbl}")

    # create output dir
    uid_obs = tbl.iloc[0]["uid_obs"]
    output_dir = Path(option.output_dir)
    ql_output_dir = output_dir.joinpath(uid_obs)

    if not ql_output_dir.exists():
        ql_output_dir.mkdir(parents=True)

    search_paths = option.search_paths or []
    search_paths.append(
        output_dir,
    )

    if option.apt_design is not None:
        apt = QTable.read(option.apt_design, format="ascii.ecsv")
    else:
        apt = _get_or_create_default_apt(output_dir.joinpath("apt_design.ecsv"))

    ql_prods = make_quicklook_prod(
        tbl,
        show_plot=option.show_plot,
        output_dir=ql_output_dir,
        search_paths=search_paths,
        apt_design=apt,
    )

    # TODO: re-enable these
    # # call Nat's vna checker
    # if tbl['file_suffix'][0] == 'vnasweep':
    #     from vnasweep_plotter_v2 import vna_rms
    #     year = 2303 #2011 #2101
    #     # testing out the class
    #     v = vna_rms(bods=bods, year=year)
    #
    #     # plotting
    #     rms_plot = 1
    #     wf_plot = 1
    #
    #     # make save path
    #     obsnum = bods.index_table['obsnum'][0]
    #     subobsnum = bods.index_table['subobsnum'][0]
    #     scannum = bods.index_table['scannum'][0]
    #     outname = f"sweepcheck_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}"
    #     # rms plot
    #     if rms_plot:
    #         ctx = v.medS21_plot()
    #         save_filepath=ql_output_dir.joinpath(outname + "_rms.png")
    #         ctx['fig'].savefig(save_filepath)
    #
    #     # noise plot at each frequency
    #     if wf_plot:
    #         # make plot
    #         ctx = v.noise_plot()
    #         save_filepath=ql_output_dir.joinpath(outname + "_noise.png")
    #         ctx['fig'].savefig(save_filepath)
    #
