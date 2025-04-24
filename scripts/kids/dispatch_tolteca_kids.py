from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, timeit, reset_logger
from tolteca_config.core import RuntimeContext
from tolteca_datamodels.toltec.file import SourceInfoModel
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_kids.core import Kids


@timeit
def reduce_sweep(rc: RuntimeContext, data):
    source_info = SourceInfoModel.model_validate(data)
    filepath = source_info.filepath
    with timeit("load kids data"):
        swp = NcFileIO(filepath).read()

    context_vars = ["roach", "file_suffix"]
    with rc.config_backend.set_context({k: swp.meta[k] for k in context_vars}):
        kids = Kids(rc)
        logger.debug(f"kids config:\n{kids.config.model_dump_yaml()}")
        try:
            kids.pipeline(swp)
        except Exception as e:
            logger.opt(exception=True).error(f"reduce sweep failed for {filepath}: {e}")
            returncode = 1
            message = "reduction failed"
        else:
            ctx = kids.kids_find.get_context(swp)
            n_chans = swp.n_chans
            n_kids = len(ctx.data.detected)
            message = f"{n_kids=} {n_chans=}"
            returncode = 0
        return returncode, locals()


def reduce_tolteca_kids(rc: RuntimeContext, data):
    # dispatch by data kind
    source_info = SourceInfoModel.model_validate(data)
    data_kind = source_info.data_kind

    if data_kind is not None and (data_kind & ToltecDataKind.RawSweep):
        logger.debug(f"reduce {data_kind=}: {source_info.filepath}")
        return reduce_sweep(rc, source_info)
    logger.debug(f"no-op for {data_kind=}: {source_info.filepath}")
    return 0, {"message": "skipped, not kids data"}


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
    LmtToltecPathOption.add_args_to_parser(parser, obs_spec_required=True)

    kids_cli_args, args = split_cli_args(r"^kids\..+", sys.argv[1:])
    logger.debug(
        f"kids_cli_args:\n{pformat_yaml(kids_cli_args)}\n"
        f"other_args:\n{pformat_yaml(args)}",
    )
    option = parser.parse_args(args)
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")
    path_option = LmtToltecPathOption(option)

    tbl = path_option.get_raw_obs_info_table(raise_on_empty=True).sort_values("roach")
    kids_cli_args[:0] = [
        "--kids.sweep_check_plot.save_path",
        path_option.dataprod_path,
        "--kids.kids_find_plot.save_path",
        path_option.dataprod_path,
        "--kids.output.path",
        path_option.dataprod_path,
        "--kids.tlaloc_output.enabled",
        "--kids.tlaloc_output.path",
        path_option.tlaloc_etc_path,
    ]
    if option.save_plot:
        kids_cli_args.extend(
            [
                "--kids.sweep_check_plot.enabled",
                "--kids.kids_find_plot.enabled",
            ],
        )

    rc = RuntimeContext(option.config)
    rc.config_backend.update_override_config(dict_from_cli_args(kids_cli_args))
    logger.info(f"{pformat_yaml(rc.config.model_dump())}")

    returncodes = []
    messages = []
    for data in tbl.toltec_file.to_info_list():
        r, ctx = reduce_tolteca_kids(rc, data)
        returncodes.append(r)
        messages.append(ctx.get("message", None))
    tbl["returncode"] = returncodes
    tbl["message"] = messages
    tbl_report = tbl[["uid_raw_obs_file", "returncode", "message"]]
    logger.info(f"run status:\n{tbl_report}")
    n_failed = np.sum(tbl["returncode"] != 0)
    if n_failed == 0:
        logger.info("Job's done!")
    else:
        logger.error("Job's failed.")
    sys.exit(n_failed)
