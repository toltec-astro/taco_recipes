import itertools
import re
from pathlib import Path

import numpy as np
from astropy.table import Column
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, timeit
from tolteca_config.core import RuntimeContext
from tolteca_datamodels.toltec.file import guess_info_from_source
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_kids.core import Kids


def _sort_targ_out_fft(swp, targ_out):
    # sort in fft order as required by the ROACH system
    lofreq = swp.meta["flo_center"]
    logger.info(f"subtract lo_freq = {lofreq}")
    dfs = targ_out["f_out"] - lofreq

    targ_out.add_column(Column(dfs, name="f_centered"), 0)
    targ_out.meta["Header.Toltec.LoCenterFreq"] = lofreq
    tones = targ_out["f_centered"]
    max_ = 3 * np.max(tones)  # this is to shift the negative tones to positive side
    isort = sorted(
        range(len(tones)), key=lambda i: tones[i] + max_ if tones[i] < 0 else tones[i]
    )
    targ_out = targ_out[isort]
    return targ_out


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

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--data_lmt_root", default="data_lmt")
    parser.add_argument("--config", default="tolteca.yaml")

    option, unparsed_args = parser.parse_known_args()
    input_ = option.input

    data_lmt_root = Path(option.data_lmt_root)
    if re.match(r"^(.*/)?toltec.+\.nc", input_):
        filepath = input_
    elif re.match(r"\d+-\d+-\d+-\d+", input_):
        obsnum, subobsnum, scannum, roach = map(int, input_.split("-"))
        p = (
            f"toltec{roach}/toltec{roach}_{obsnum:06d}"
            f"_{subobsnum:03d}_{scannum:04d}_*.nc"
        )
        glob_patterns = [
            f"toltec/ics/{p}",
            f"toltec/tcs/{p}",
        ]
        files = list(itertools.chain(*(data_lmt_root.glob(p) for p in glob_patterns)))
        if len(files) != 1:
            raise ValueError("ambiguous input file.")
        filepath = files[0]
    else:
        raise ValueError(f"invalid input {input_}")

    logger.info(f"{filepath=}")
    rc = RuntimeContext(option.config)

    from tollan.utils.cli import dict_from_cli_args

    rc.config_backend.update_override_config(dict_from_cli_args(unparsed_args))
    logger.info(f"{pformat_yaml(rc.config.model_dump())}")

    source_info = guess_info_from_source(filepath)
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