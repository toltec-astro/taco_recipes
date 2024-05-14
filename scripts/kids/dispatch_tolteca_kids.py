from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, timeit, reset_logger
from tolteca_config.core import RuntimeContext
from tolteca_datamodels.toltec.file import guess_info_from_source
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_kids.core import Kids


@timeit
def reduce_sweep_tolteca_kids(rc: RuntimeContext, filepath):
    with timeit("load kids data"):
        swp = NcFileIO(filepath).read()

    context_vars = ["roach", "file_suffix"]
    with rc.config_backend.set_context({k: swp.meta[k] for k in context_vars}):
        kids = Kids(rc)
        logger.debug(f"kids config:\n{kids.config.model_dump_yaml()}")
        try:
            kids.pipeline(swp)
        except Exception:
            logger.opt(exception=True).error("failed to run tolteca kids:")
            returncode = -1
        else:
            returncode = 0
        return locals()


if __name__ == "__main__":
    import argparse
    from toltec_file_utils import LmtToltecPathOption
    from tollan.utils.cli import dict_from_cli_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tolteca.yaml")
    parser.add_argument("--log_level", default="INFO", help="The log level.")
    LmtToltecPathOption.add_args_to_parser(parser, obs_spec_required=True)
    option, unparsed_args = parser.parse_known_args()
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")
    path_option = LmtToltecPathOption(option)

    filepath = path_option.get_raw_obs_files(unique=True)
    if filepath is None:
        raise ValueError(f"no files found for obs_spec={option.obs_spec}")
    source_info = guess_info_from_source(filepath)
    logger.debug(
        f"resolved info for {filepath=}\n{pformat_yaml(source_info.model_dump())}"
    )

    rc = RuntimeContext(option.config)
    rc.config_backend.update_override_config(dict_from_cli_args(unparsed_args))
    logger.info(f"{pformat_yaml(rc.config.model_dump())}")

    # dispatch by data kind
    data_kind = source_info.data_kind
    if source_info.data_kind & ToltecDataKind.RawSweep:
        ctx = reduce_sweep_tolteca_kids(rc, filepath)
        if ctx["returncode"] < 0:
            logger.error("Job's failed.")
        else:
            logger.info("Job's done!")
    else:
        logger.debug(f"no-op for {data_kind=}")
