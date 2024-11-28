import sys
from pathlib import Path
from typing import Literal, get_args, ClassVar
import numpy as np

from tollan.utils.general import ensure_abspath
from tollan.utils.log import logger, reset_logger, timeit
from tollan.utils.sys import pty_run
from tolteca_datamodels.toltec.file import guess_info_from_source
from toltec_file_utils import LmtToltecPathOption


taco_recipe_script_dir = Path(__file__).parent.parent


kids_script_dir = taco_recipe_script_dir.joinpath("kids")


def run_tolteca_kids(
    obs_spec,
    unparsed_args,
    data_lmt_path,
    etc_path,
    log_level="INFO",
    **kwargs,
):
    output_path = data_lmt_path.joinpath("toltec/reduced")
    cmd = (
        [
            "bash",
            kids_script_dir.joinpath("dispatch_tolteca_kids.sh"),
            "--log_level",
            log_level,
            "--data_lmt_path",
            data_lmt_path,
            "--tlaloc_etc_path",
            etc_path,
            "--kids.output.enabled",
            "--kids.output.path",
            output_path,
            "--kids.output.subdir_fmt",
            "null",
            "--kids.tlaloc_output.enabled",
        ]
        + unparsed_args
        + ["--", obs_spec]
    )
    return pty_run(cmd)


drivefit_script_dir = taco_recipe_script_dir.joinpath("drivefit")


def run_drivefit(
    mode,
    obs_spec,
    unparsed_args,
    data_lmt_path,
    etc_path,
    log_level="INFO",
    **kwargs,
):
    output_path = data_lmt_path.joinpath("toltec/reduced")
    cmd = (
        [
            "bash",
            drivefit_script_dir.joinpath("dispatch_drivefit.sh"),
            "--log_level",
            log_level,
            "--data_lmt_path",
            data_lmt_path,
            "--tlaloc_etc_path",
            etc_path,
            "--dataprod_path",
            output_path,
            "--mode",
            mode,
            "--select",
            "file_suffix=='targsweep'",
        ]
        + unparsed_args
        + ["--", obs_spec]
    )
    return pty_run(cmd)


def run_drivefit_old(tbl):
    returncodes = []
    for source_info in tbl.toltec_file.to_info_list():
        cmd = [
            "bash",
            drivefit_script_dir.joinpath("reduce_drivefit.sh"),
            source_info.obsnum,
            source_info.roach,
        ]
        returncodes.append(pty_run(cmd))
    return np.sum(returncodes)


def run_drivefit_commit_old(tbl, etc_dir="~/tlaloc/etc"):
    if len(tbl) > 1:
        logger.error("drivefit commit for multiple files is not implemented yet.")
        return 1
    # TODO: properly handle the multiple file case
    source_info = tbl.to_info_list()[0]
    path = source_info.filepath
    # find drive fit obsnum by going back for
    nw = source_info.roach
    obsnum = source_info.obsnum
    data_dir = path.parent
    # TODO: this assumes the drive fit commit always
    # comes from the drive fit data of the same master
    # which is not true. Need properly handle the searching
    # via data product database.
    n_obsnums_lookback = 100
    for o in range(obsnum, obsnum - n_obsnums_lookback, -1):
        pattern = f"toltec{nw}_{o:06d}_*_*targsweep.nc"
        targ_files = list(data_dir.glob(pattern))
        logger.debug(f"glob {data_dir}/{pattern}")
        if targ_files:
            logger.debug(f"found {len(targ_files)} targ sweep files for obsnum={o}")
        if len(targ_files) >= 5:
            source_info_drivefit = guess_info_from_source(targ_files[0])
            break
    else:
        source_info_drivefit = None
    if source_info_drivefit is None:
        logger.info("unable to locate drivefit obsnum, abort.")
        return 1
    etc_dir = ensure_abspath(etc_dir)
    cmd = [
        "bash",
        drivefit_script_dir.joinpath("reduce_drivefit_commit_local.sh"),
        source_info_drivefit.obsnum,
        source_info.obsnum,
        source_info.roach,
        etc_dir,
    ]
    return pty_run(cmd)


TlalocActionType = Literal[
    "vna_reduce",
    "targ_reduce",
    "timestream_reduce",
    "targ_reduce_bg",
    "drivefit_reduce",
    "drivefit_commit",
    "dry_run",
]


class TlalocAction:
    """The tlaloc action."""

    actions: ClassVar[list[TlalocActionType]] = list(get_args(TlalocActionType))

    @classmethod
    def vna_reduce(cls, *args, **kwargs):
        return run_tolteca_kids(*args, **kwargs)

    @classmethod
    def targ_reduce(cls, *args, **kwargs):
        return run_tolteca_kids(*args, **kwargs)

    @classmethod
    def timestream_reduce(cls, *args, **kwargs):
        return run_tolteca_kids(*args, **kwargs)

    @classmethod
    def drivefit_reduce(cls, *args, **kwargs):
        return run_drivefit("drivefit", *args, **kwargs)

    @classmethod
    def drivefit_commit(cls, *args, **kwargs):
        return run_drivefit("drivefit_commit", *args, **kwargs)

    @classmethod
    def dry_run(cls, *args, **kwargs):
        logger.info(f"DRY RUN:\n{args=}\n{kwargs=}")
        return 0

    @classmethod
    def run(cls, action, *args, **kwargs) -> int:
        with timeit(f"run tlaloc {action=}", level="INFO"):
            return getattr(cls, action)(*args, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dispatch kids related calls from tlaloc."
    )
    parser.add_argument("--log_level", default="DEBUG", help="The log level.")
    g = parser.add_mutually_exclusive_group()

    g.add_argument(
        "--fg_plot",
        "-f",
        action="store_true",
        help="No-op, existing for compat purpose.",
    )
    g.add_argument(
        "--bg_plot",
        "-b",
        action="store_true",
        help="No-op, existing for compat purpose.",
    )
    g.add_argument(
        "--reduce_only",
        "-r",
        action="store_true",
        help="No-op, existing for compat purpose.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="No-op, existing for compat purpose.",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=TlalocAction.actions,
        help="The tlaloc action.",
    )
    parser.add_argument(
        "--etc_path",
        default="~/tlaloc/etc",
        type=ensure_abspath,
        help="data_lmt path",
    )
    LmtToltecPathOption.add_data_lmt_path_argument(parser)
    LmtToltecPathOption.add_obs_spec_argument(parser)

    option, unparsed_args = parser.parse_known_args()
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")

    # here we need to expand the obsspec so they collect all files for
    # the obsnum
    if option.action == "drivefit_reduce":
        path_option = LmtToltecPathOption(option)
        tbl = path_option.get_raw_obs_info_table(raise_on_empty=True)
        entry = next(tbl.itertuples())
        obs_spec = f"{entry.obsnum}/{entry.roach}" 
        logger.info(f"resolved obsspec for drivefit {option.obs_spec} -> {obs_spec}")
    else:
        obs_spec = option.obs_spec


    sys.exit(
        TlalocAction.run(
            option.action,
            obs_spec,
            unparsed_args=unparsed_args,
            data_lmt_path=option.data_lmt_path,
            etc_path=option.etc_path,
            log_level=option.log_level,
        )
    )
