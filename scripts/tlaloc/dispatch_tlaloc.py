import sys
from pathlib import Path
from typing import Literal, get_args, ClassVar

from tollan.utils.general import ensure_abspath
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, reset_logger, timeit
from tollan.utils.sys import pty_run
from tolteca_datamodels.toltec.file import guess_info_from_source, SourceInfoModel
from toltec_file_utils import LmtToltecPathOption


taco_recipe_script_dir = Path(__file__).parent.parent


def _ensure_source_info(data):
    if isinstance(data, SourceInfoModel):
        return data
    return guess_info_from_source(data)


kids_script_dir = taco_recipe_script_dir.joinpath("kids")


def run_tolteca_kids(data):
    source_info = _ensure_source_info(data)
    cmd = [
        "bash",
        kids_script_dir.joinpath("dispatch_tolteca_kids.sh"),
        source_info.filepath,
    ]
    return pty_run(cmd)


drivefit_script_dir = taco_recipe_script_dir.joinpath("drivefit")


def run_drivefit(data):
    source_info = _ensure_source_info(data)
    cmd = [
        "bash",
        drivefit_script_dir.joinpath("reduce_drivefit.sh"),
        source_info.obsnum,
        source_info.roach,
    ]
    return pty_run(cmd)


def run_drivefit_commit(data, etc_dir="~/tlaloc/etc"):
    source_info = _ensure_source_info(data)
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
    def vna_reduce(cls, data, **kwargs):
        return run_tolteca_kids(data)

    @classmethod
    def targ_reduce(cls, data, **kwargs):
        return run_tolteca_kids(data)

    @classmethod
    def timestream_reduce(cls, data, **kwargs):
        return run_tolteca_kids(data)

    @classmethod
    def drivefit_reduce(cls, data, **kwargs):
        return run_drivefit(data)

    @classmethod
    def drivefit_commit(cls, data, etc_dir, **kwargs):
        return run_drivefit_commit(data, etc_dir=etc_dir)

    @classmethod
    def dry_run(cls, data, **kwargs):
        logger.info(f"DRY RUN: {data=} {kwargs=}")
        return 0

    @classmethod
    def run(cls, action, data, **kwargs) -> int:
        with timeit(f"run tlaloc {action=}", level="INFO"):
            return getattr(cls, action)(data, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dispatch kids related calls from tlaloc."
    )
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
        help="The tlaloc output directory.",
    )
    parser.add_argument("--log_level", default="INFO", help="The log level.")

    parser.add_argument(
        "--action",
        required=True,
        choices=TlalocAction.actions,
        help="The tlaloc action.",
    )
    LmtToltecPathOption.add_args_to_parser(parser, obs_spec_required=True)
    option = parser.parse_args()
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")

    # TODO: may need to update tlaloc to make this better describing its purpose.
    output = option.output
    if output is not None:
        if output.is_dir():
            etc_dir = output
        else:
            etc_dir = output.parent.parent
    else:
        etc_dir = ensure_abspath("~/tlaloc/etc")
    logger.debug(f"resolved {etc_dir=}")

    path_option = LmtToltecPathOption(option, tlaloc_etc_path=etc_dir)
    filepath = path_option.get_raw_obs_files(unique=True)
    if filepath is None:
        raise ValueError(f"no files found for obs_spec={option.obs_spec}")
    source_info = guess_info_from_source(filepath)
    logger.debug(
        f"resolved info for {filepath=}\n{pformat_yaml(source_info.model_dump())}"
    )
    sys.exit(TlalocAction.run(option.action, data=source_info, etc_dir=etc_dir))
