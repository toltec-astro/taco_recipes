import sys
import pty
from tollan.utils.log import logger
from tollan.utils.fmt import pformat_yaml
from tolteca_datamodels.toltec.file import guess_meta_from_source
from pathlib import Path


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
        "--scratch_dir", "-d", type=Path, help="Output dir for intermediate files."
    )
    parser.add_argument("--output", "-o", type=Path, help="Output file path.")
    parser.add_argument("--log_level", default="INFO", help="The log level.")


    action_enum = {
            "vna_reduce": 0,
            "targ_reduce": 1,
            "timestream_reduce": 2,
            "targ_reduce_bg": 3,
            "drivefit_reduce": 4,
            "drivefit_commit": 5,
            }

    parser.add_argument(
        "--action", required=True,
        choices=list(action_enum.keys()),
        help="The action."
    )
    parser.add_argument("filepath", type=Path, help="The file to process.")

    option = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=option.log_level)

    logger.debug(f"parsed options: {option}")

    filepath = option.filepath
    for _ in range(5):
        if not filepath.is_symlink():
            break
        _filepath = filepath
        filepath = _filepath.readlink()
        if not filepath.is_absolute():
            filepath = (_filepath.parent / filepath).resolve()
    logger.info(f"resolved link: {filepath}")

    meta = guess_meta_from_source(filepath)

    logger.debug(f"{filepath=}\n{pformat_yaml(meta)}")


    script_dir = Path(__file__).parent
    action = option.action
    if action == 'drivefit_reduce':
        cmd = list(map(str, [
            script_dir.joinpath("reduce_drivefit.sh").as_posix(),
            meta["obsnum"],
            meta["roach"],
            ]))
        logger.info("run {}".format(" ".join(cmd)))
        returncode = pty.spawn(cmd)
        logger.info(f"command {returncode=}")

    if action == 'drivefit_commit':
        # find drive fit obsnum
        nw = meta["roach"]
        obsnum = meta["obsnum"]
        data_dir = filepath.parent
        for o in range(obsnum, obsnum - 100, -1):
            pattern = f"toltec{nw}_{o:06d}_*_*targsweep.nc"
            targ_files = list(
                    data_dir.glob(
                        pattern)
                    )
            logger.debug(f"glob {data_dir}/{pattern}")
            if targ_files:
                logger.debug(f"{targ_files}")
            if len(targ_files) >= 5:
                dmeta = guess_meta_from_source(targ_files[0])
                break
        else:
            dmeta = None
        if dmeta is None:
            logger.info("unable to locate drivefit obsnum, abort.")
            sys.exit(1)
        output = option.output
        if output is not None:
            if output.is_dir():
                etc_dir = output
            else:
                etc_dir = output.parent.parent
        else:
            etc_dir = Path("/home/toltec/tlaloc/etc/")
        cmd = list(map(str, [
            script_dir.joinpath("reduce_drivefit_commit_local.sh").as_posix(),
            dmeta["obsnum"],
            meta["obsnum"],
            meta["roach"],
            etc_dir
            ]))
        logger.info("run {}".format(" ".join(cmd)))
        returncode = pty.spawn(cmd)
        logger.info(f"command {returncode=}")
