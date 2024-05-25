from __future__ import annotations
from tollan.utils.general import rupdate
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, timeit, reset_logger
from tolteca_config.core import RuntimeContext
import pandas as pd
from tollan.utils.plot.plotly import make_range
from tolteca_config.core import ConfigHandler, ConfigModel, SubConfigKeyTransformer
from typing import Literal, TYPE_CHECKING
from pydantic import Field
from tollan.config.types import AbsDirectoryPath
from tolteca_kids.plot import PlotMixin, PlotConfig

if TYPE_CHECKING:
    from tolteca_datamodels.toltec.file import SourceInfoDataFrame


class DriveFitConfig(ConfigModel):
    """The config for drive fit."""

    n_adrvs_min: int = Field(
        default=5,
        description="Minimum number of drive atten values required.",
    )
    output_path: AbsDirectoryPath = Field(
        description="Output path.",
    )
    save_plot: bool = Field(default=False, description="Save plots.")
    weight_window_Qr: float = Field(
        default=12000.0,
        description="Use weighting window of specfiied Qr.",
    )
    a_ref: float = Field(
        default=0.2,
        description="Non-linearity parameter to use for deriving best driving power.",
    )
    a_min: float = Field(
        default=-0.2,
        description="Minimum a value used for a_best determination.",
    )
    a_max: float = Field(
        default=1.5,
        description="Maximum a value used for a_best determination.",
    )


class DriveFit(
    SubConfigKeyTransformer[Literal["drivefit"]], ConfigHandler[DriveFitConfig]
):
    """The class to perform drivefit."""

    _adrv_meta_key = "atten_drive"

    @classmethod
    def validate_inputs(cls, tbl: SourceInfoDataFrame, n_adrvs_min=5):
        """Return validated inputs for drive fit run."""
        stbl = tbl.query(
            "file_suffix in ['tune', 'targsweep'] & file_ext == '.nc'"
        ).copy()
        if len(stbl) == 0:
            raise ValueError("no sweep data found in inputs.")
        stbl = stbl.toltec_file.read()
        adrv_meta_key = cls._adrv_meta_key
        logger.debug(
            f"{len(stbl)} of {len(tbl)} sweep data found in inputs\n"
            f"{stbl.toltec_file.pformat(type='short', include_cols=[adrv_meta_key])}"
        )
        # check channels are the same
        # TODO: implement more stringent check to ensure the channel frequencies
        # are matched

        n_chans_unique = np.unique(stbl["n_chans"])
        if len(n_chans_unique) > 1:
            raise ValueError(f"inconsistent sweep channels: {n_chans_unique=}")

        roach_unique = np.unique(stbl["roach"])
        if len(roach_unique) > 1:
            raise ValueError(f"inconsistent roach: {roach_unique=}")

        adrvs = np.unique(stbl[adrv_meta_key])
        n_adrvs = len(adrvs)
        if n_adrvs < n_adrvs_min:
            raise ValueError(f"too few atten_drive steps: {n_adrvs=} {n_adrvs_min=}")
        n_chans = n_chans_unique[0]
        roach = roach_unique[0]
        logger.debug(f"validate drivefit input: {roach=} {n_chans=} {n_adrvs=}\n")
        return stbl

    def __call__(self, tbl: SourceInfoDataFrame):
        """Run drive fit for given dataset."""
        cfg = self.config
        tbl = self.validate_inputs(tbl, n_adrvs_min=cfg.n_adrvs_min)
        return self.drivefit(tbl)

    @timeit
    def drivefit(self, tbl: SourceInfoDataFrame):
        obsnum = tbl["obsnum"].min()
        roach = tbl["roach"].min()

        cfg = self.config
        output_path = cfg.output_path
        plot_save_path = output_path.joinpath(f"{obsnum}")

        kpf_cfg = {
            "load": {
                "save_dir": output_path,
                "fig_dir": plot_save_path,
            },
            "save": {
                "use_save_fig": False,
                "use_save_pdf": False and cfg.save_plot,
                "use_save_file": True,
                "save_name": "adrv",
            },
            "preview": {
                "show_plots": False,
            },
            "weight": {
                "window_Qr": cfg.weight_window_Qr,
                "weight_type": "lorentz",
            },
            "flag_settings": {
                "a_predict_guess": 0.10,
                "a_predict_threshold": 0.25,
                "pherr_threshold": 0.25,
                "pherr_threshold_num": 10,
            },
            "fit_settings": {
                "powlist_start": None,
                "powlist_end": None,
                "numspan": 1,
                "tone_range": list(range(100, 120)),
                # "tone_range": "all",
            },
        }
        from kid_phase_fit import do_toltec_drive_fit

        kpf_ctx = do_toltec_drive_fit(kpf_cfg, roach, obsnum, tbl["filepath"])
        chan_indices = kpf_ctx["tone_range"]
        all_fits = kpf_ctx["all_fits"]

        # extract fit info and interpolate to get best drivefit value
        drivefit_ctxs = {}
        for ci in chan_indices:
            fit_obj = all_fits[ci]
            drivefit_ctxs[ci] = self._find_best_adrv(fit_obj, a_ref=cfg.a_ref)
        drivefit_flags = np.array([drivefit_ctxs[ci]["p_flag"] for ci in chan_indices])
        n_success = np.sum(drivefit_flags == 0)
        n_chans = len(chan_indices)
        result = locals()
        if cfg.save_plot:
            self.plot_drivefit(result)
        return result

    @classmethod
    def _find_best_adrv(
        cls,
        fit_obj,
        a_ref=DriveFitConfig.model_fields["a_ref"].default,
        a_min=DriveFitConfig.model_fields["a_min"].default,
        a_max=DriveFitConfig.model_fields["a_max"].default,
        a_tolerance=0.01,
    ):
        """This function determining the best driving power empirically."""
        a_values = np.array(fit_obj.bif_list)
        pows = np.array(fit_obj.powlist)
        fit_flags = np.array(fit_obj.fit_flag_list)
        m = (fit_flags == 0) & (a_values <= a_max) & (a_values >= a_min)
        n_interp = m.sum()
        p_interp = pows[m]
        a_interp = a_values[m]
        if n_interp == 0:
            p_best = None
            a_best = None
            p_flag = 1
            logger.debug(
                "no valid data point for finding best atten drive.",
            )
            return locals()
        if n_interp < 2:
            # just get the only data point
            p_best = p_interp[0]
            a_best = a_interp[0]
            p_flag = 1
            logger.debug(
                f"only one data point found, used as best atten drive {p_best=}",
            )
            return locals()
        # sort by p and check zero crossing for nonlinearity
        isort = np.argsort(p_interp)
        p_sorted = p_interp[isort]
        a_sorted = a_interp[isort]
        p_best, y, p_flag = cls._find_first_zerocrossing1d(
            p_sorted, a_sorted - a_ref, tolerance=a_tolerance
        )
        a_best = y + a_ref
        logger.debug(
            f"use {n_interp} data points, found {p_best=} {a_best=} {p_flag=}",
        )
        return locals()

    @staticmethod
    def _find_first_zerocrossing1d(x, y, tolerance=1e-5):
        # no crossing and all positive, return the first one
        if y[0] >= 0:
            return x[0], y[0], 1
        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            x0, x1 = x[i], x[i + 1]
            if np.isclose(y0, 0, atol=tolerance):
                return x0, y0, 0
            if y0 < 0 and y1 >= 0:
                # interpolate
                dx = x1 - x0
                dy = y1 - y0
                return x0 - y0 * dx / dy, 0.0, 0
        else:
            # no crossing and all negative, return the last one
            return x[-1], y[-1], 1

    @timeit
    def plot_drivefit(self, ctx):
        """Make plot to show the fitting."""
        fig = self._make_drivefit_fig(ctx)

        save_path = ctx["plot_save_path"]
        roach = ctx["roach"]
        obsnum = ctx["obsnum"]
        if not save_path.exists():
            save_path.mkdir(parents=True)
        save_name = f"toltec{roach}_{obsnum:06d}_drivefit_summary.html"
        PlotConfig.save_plotly_fig(save_path.joinpath(save_name), fig)

    def _make_drivefit_fig(self, ctx):
        drivefit_ctxs = ctx["drivefit_ctxs"]
        chan_indices = ctx["chan_indices"]
        n_chans = len(chan_indices)
        pows = next(iter(drivefit_ctxs.values()))["pows"]
        n_pows = len(pows)

        n_rows = 7
        n_cols = n_pows + 1

        color_cycle = PlotMixin.color_palette.cycle()

        def logmag(z):
            return 20.0 * np.log10(np.abs(z))

        def _make_drivefit_panel(row, col, panel_id, dummy=False, init=False):
            i = panel_id // n_cols
            pi = panel_id % n_cols
            if pi == n_cols - 1:
                # last panel is the adrv summary plot
                pi = None
            if i >= n_chans:
                x = []
                y = []
                color = None
                name = None
            else:
                ci = chan_indices[i]
                drivefit_ctx = drivefit_ctxs[ci]
                p_best = drivefit_ctx["p_best"]
                a_best = drivefit_ctx["a_best"]
                fit_obj = drivefit_ctx["fit_obj"]
                color = next(color_cycle)
                pp = pows[pi]
                if pi is not None:
                    if pp in fit_obj.z1:
                        x = fit_obj.f[pp]
                        y = logmag(fit_obj.z1[pp])
                    else:
                        x = []
                        y = []
                else:
                    x = drivefit_ctx["p_interp"]
                    y = drivefit_ctx["a_interp"]
                name = f"chan {ci} pow {pi}"
            trace = {
                "type": "scatter",
                "x": x,
                "y": y,
                "name": name,
                "marker": {
                    "color": color,
                },
            }
            if init:
                rupdate(
                    trace,
                    {
                        "mode": "markers",
                        "marker": {
                            "size": 12 if pi is None else 4,
                        },
                    },
                )
            data = [trace]
            # add best adrv values in summary panel
            if i < n_chans and pi is None and p_best is not None:
                data.append(
                    {
                        "type": "scatter",
                        "x": [p_best],
                        "y": [a_best],
                        "mode": "markers",
                        "marker": {
                            "size": 16,
                            "symbol": "x",
                            "color": "rgba(0, 0, 0, 0)",
                            "line": {"color": "red", "width": 2},
                        },
                    }
                )
            # text annos
            if i < n_chans:
                if pi is None:
                    text_lns = [
                        f"p_flag={drivefit_ctx['p_flag']}",
                    ]
                    if p_best is not None and a_best is not None:
                        text_lns.extend(
                            [
                                f"a_best={drivefit_ctx['a_best']:.2f}",
                                f"p_best={drivefit_ctx['p_best']:.2f}",
                            ]
                        )
                else:
                    text_lns = []
                    if row == 1:
                        text_lns.append(f"pow: {pows[pi]}")
                    if col == 1:
                        text_lns.append(f"chan: {ci}")
                    if pp in fit_obj.result:
                        fit_flag = fit_obj.fit_flag[pp]
                        a = fit_obj.result[pp].result["bif"]
                        text_lns.extend(
                            [
                                f"fit_flag: {fit_flag}",
                                f"a: {a:.2g}",
                            ]
                        )
                    else:
                        text_lns.append("fit_flag: 1")
                if text_lns:
                    annos = [
                        {
                            "x": 0.4,
                            "y": 0.8,
                            "xref": "x domain",
                            "yref": "y domain",
                            "text": "<br>".join(text_lns),
                            "align": "left",
                            "valign": "top",
                        }
                    ]
                else:
                    annos = []
            else:
                annos = []
            if dummy:
                xrange = [0, 1]
                yrange = [0, 1]
            else:
                x = data[0]["x"]
                y = data[0]["y"]
                if len(x) == 0:
                    xrange = [0, 1]
                    yrange = [0, 1]
                else:
                    xrange = make_range(data[0]["x"])
                    yrange = make_range(data[0]["y"])
            layout = {
                "xaxis": {
                    "range": xrange,
                },
                "yaxis": {
                    "range": yrange,
                },
            }
            if init:
                if row == n_rows and col == 1:
                    rupdate(
                        layout,
                        {
                            "xaxis": {
                                "title": {
                                    "text": "f (Hz)",
                                },
                            },
                            "yaxis": {
                                "title": {
                                    "text": "S21 (dB)",
                                },
                            },
                        },
                    )
                if row == n_rows and col == n_cols:
                    rupdate(
                        layout,
                        {
                            "xaxis": {
                                "title": {
                                    "text": "Drive Power (dB)",
                                },
                            },
                            "yaxis": {
                                "title": {
                                    "text": "Non-linearity",
                                },
                            },
                        },
                    )
            rupdate(
                layout,
                {
                    "annotations": annos,
                },
            )
            return {
                "data": data,
                "layout": layout,
            }

        n_items = n_chans * n_cols
        fig = PlotMixin.make_data_grid_anim(
            name="Drive Fit",
            n_rows=n_rows,
            n_cols=n_cols,
            n_items=n_items,
            make_panel_func=_make_drivefit_panel,
            frame_name_func=lambda s, e: chan_indices[s // n_cols],
            redraw=False,
            subplot_titles=None,
        )
        fig.update_layout(
            margin={
                "t": 100,
                "b": 100,
            }
        )
        return fig


def run_drivefit(rc: RuntimeContext, tbl: SourceInfoDataFrame):
    roach_unique = np.unique(tbl["roach"])
    if len(roach_unique) > 1:
        message = "inconsistent roach"
        return 1, locals()
    roach = roach_unique[0]
    with rc.config_backend.set_context({"roach": roach}):
        drivefit = DriveFit(rc)
        logger.debug(f"drivefit config:\n{drivefit.config.model_dump_yaml()}")
        try:
            ctx = drivefit(tbl)
        except Exception as e:
            logger.opt(exception=True).error(f"drivefit failed for {roach=}: {e}")
            returncode = 1
            message = f"run failed: {e}"
        else:
            n_success = ctx["n_success"]
            n_chans = ctx["n_chans"]
            message = f"{n_success=} {n_chans=}"
            returncode = 0
        return returncode, locals()


if __name__ == "__main__":
    import sys
    import argparse
    import numpy as np
    from toltec_file_utils import LmtToltecPathOption
    from tollan.utils.cli import dict_from_cli_args, split_cli_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tolteca.yaml")
    parser.add_argument("--log_level", default="INFO", help="The log level.")
    parser.add_argument("--save_plot", action="store_true", help="Save plots.")
    LmtToltecPathOption.add_args_to_parser(
        parser, obs_spec_required=True, obs_spec_multi=True
    )

    drivefit_cli_args, args = split_cli_args("^drivefit\..+", sys.argv[1:])
    logger.debug(
        f"drivefit_cli_args:\n{pformat_yaml(drivefit_cli_args)}\n"
        f"other_args:\n{pformat_yaml(args)}",
    )
    option = parser.parse_args(args)
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")
    path_option = LmtToltecPathOption(option)

    tbl = path_option.get_raw_obs_info_table(raise_on_empty=True).sort_values("roach")
    drivefit_cli_args[:0] = [
        "--drivefit.output_path",
        path_option.dataprod_path,
    ]

    if option.save_plot:
        drivefit_cli_args.extend(
            [
                "--drivefit.save_plot",
            ],
        )
    rc = RuntimeContext(option.config)
    rc.config_backend.update_override_config(dict_from_cli_args(drivefit_cli_args))
    logger.info(f"{pformat_yaml(rc.config.model_dump())}")

    with timeit("run drive fit"):
        report = []
        for roach, stbl in tbl.groupby("roach", sort=False):
            r, ctx = run_drivefit(rc, stbl)
            report.append(
                {
                    "roach": roach,
                    "n_items": len(stbl),
                    "returncode": r,
                    "message": ctx.get("message", None),
                }
            )
    report = pd.DataFrame.from_records(report)
    logger.info(f"run status:\n{report}")
    n_failed = np.sum(report["returncode"] != 0)
    if n_failed == 0:
        logger.info("Job's done!")
    else:
        logger.error("Job's failed.")
    sys.exit(n_failed)
