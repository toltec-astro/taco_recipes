from pathlib import Path
from tolteca_datamodels.lmt.filestore import LmtFileStore
import itertools
from tollan.utils.log import logger
import re
from types import SimpleNamespace
import argparse
from tollan.utils.general import ensure_abspath, dict_from_regex_match, resolve_symlink
from tolteca_datamodels.toltec.file import (
    guess_info_from_sources,
    SourceInfoModel,
)

# from tolteca_datamodels.data_prod.filestore import DataProdFileStore
from tollan.utils.fmt import pformat_yaml
from tollan.utils.nc import ncstr
import netCDF4

# from tolteca_datamodels.toltec.filestore import ToltecFileStore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tolteca_datamodels.toltec.file import SourceInfoDataFrame

source_info_fields = list(SourceInfoModel.model_fields.keys())


# def get_toltec_raw_obs_info(toltecfs, obsnum=None):
#     if obsnum is None:
#         try:
#             tbl = toltecfs.get0k_symlink_info_table()
#         except Exception as e:
#             logger.opt(exception=True).error(f"error to infer current obs info: {e}")
#             return None
#         if tbl is None:
#             logger.error(
#                 "unable to interfer current obs info: no symlink in data root."
#             )
#             return None
#         tbl = guess_info_from_sources(tbl["path"])
#     else:
#         obsnum = option.obsnum
#         files = toltecfs.glob(f"toltec*_{obsnum:06d}_*.nc")
#         tbl = guess_info_from_sources(files)
#     info = tbl.toltec_file.get_latest()
#     logger.debug(f"resolved info:{pformat_yaml(info.model_dump())}")
#     return info
#


def get_lmt_info_tbl_from_toltec_info(lmt_fs: LmtFileStore, info: SourceInfoModel):
    master = info.filepath.parent.parent.name
    logger.debug(f"inferred {master=} for {info=}")
    if master != "tcs":
        logger.error(f"no tel file for {master=} {info=}")
        return None
    glob_pattern = f"tel_*{info.obsnum:06d}_{info.subobsnum:02d}_{info.scannum:04d}.nc"
    files = list(lmt_fs.tel_path.glob(glob_pattern))
    if not files:
        logger.error(f"no tel file found for {glob_pattern=}")
        return None
    tbl = guess_info_from_sources(files)
    logger.debug(f"resolved info table:\n{tbl}")
    return tbl


def get_apt_filename(info: SourceInfoModel):
    # TODO: make rule for apt file.
    stem = info.source.path.stem.split("_", 1)[-1]
    return f"apt_{stem}.ecsv"


def get_obs_goal(lmt_fs, info: SourceInfoModel):
    if lmt_fs is None:
        logger.error("no valid data_lmt path.")
        return None
    tel_info_tbl: SourceInfoDataFrame = get_lmt_info_tbl_from_toltec_info(lmt_fs, info)
    if len(tel_info_tbl) == 0:
        logger.error("no tel data found.")
        return None
    if len(tel_info_tbl) > 1:
        logger.error("ambiguous tel data.")
    tel_info = tel_info_tbl.toltec_file.to_list()[0]
    tel_filepath = tel_info.source.path
    tel_nc = netCDF4.Dataset(tel_filepath)
    obs_goal = ncstr(tel_nc.variables["Header.Dcs.ObsGoal"]).lower()
    return obs_goal


class LmtToltecPathOption:

    def __init__(self, option, **kwargs):
        _option = SimpleNamespace()
        _option.__dict__.update(option.__dict__ | kwargs)
        option = self._option = _option
        logger.debug(f"create path option from {option=}")
        lmt_fs, toltec_fs = self._get_lmt_toltec_file_store(option)
        self._lmt_fs = lmt_fs
        self._toltec_fs = toltec_fs
        self._dp_path = self._get_dataprod_file_store(option)
        self._tlaloc_etc_path = self._get_tlaloc_etc_file_store(option)

    @classmethod
    def add_args_to_parser(
        cls, parser: argparse.ArgumentParser, defaults=None, obs_spec_required=False
    ):
        defaults = defaults or {}
        parser.add_argument(
            "--data_lmt_path",
            type=Path,
            default=defaults.get("data_lmt_path", "/data_lmt"),
            help="data_lmt path.",
        )
        parser.add_argument(
            "--dataprod_path",
            type=Path,
            default=defaults.get("dataprod_path", "/dataprod_toltec"),
            help="data product path.",
        )
        parser.add_argument(
            "--tlaloc_etc_path",
            type=Path,
            default=defaults.get("tlaloc_etc_path", ensure_abspath("/tlaloc_etc")),
            help="tlaloc etc path.",
        )
        if obs_spec_required:
            obs_spec_kw = {}
        else:
            obs_spec_kw = {
                "nargs": "?",
                "default": None,
            }
        parser.add_argument(
            "obs_spec",
            help="specifier of obs data.",
            **obs_spec_kw,
        )

    @property
    def lmt_fs(self):
        """The LMT file store."""
        return self._lmt_fs

    @property
    def toltec_fs(self):
        """The TolTEC file store."""
        return self._toltec_fs

    @property
    def dataprod_path(self):
        """The TolTEC file store."""
        return self._dp_path

    @property
    def tlaloc_etc_path(self):
        """The tlaloc etc file store."""
        return self._tlaloc_etc_path

    @classmethod
    def _get_lmt_toltec_file_store(cls, option):
        data_lmt_path = option.data_lmt_path
        logger.debug(f"{data_lmt_path=}")
        if not data_lmt_path.exists():
            logger.error(f"data_lmt path does not exit: {data_lmt_path}")
            lmt_fs = None
        else:
            lmt_fs = LmtFileStore(path=data_lmt_path)
            logger.debug(f"{lmt_fs=}")
        if lmt_fs is None:
            toltec_fs = None
        else:
            toltec_fs = lmt_fs.toltec
            logger.debug(f"{toltec_fs=}")
        return lmt_fs, toltec_fs

    @classmethod
    def _get_dataprod_file_store(cls, option):
        # TODO: add DataProdFileStore handling
        return option.dataprod_path

    @classmethod
    def _get_tlaloc_etc_file_store(cls, option):
        # TODO: add tlaloc etc file store handling
        return option.tlaloc_etc_path

    def get_raw_obs_files(self, unique=False):
        option = self._option
        obs_spec = option.obs_spec
        logger.debug(f"resovle {obs_spec=}")
        if obs_spec is None:
            toltec_fs = self.toltec_fs
            if toltec_fs is None:
                logger.error("no toltec data path specified.")
                return None
            try:
                tbl = toltec_fs.get_symlink_info_table()
            except Exception as e:
                logger.opt(exception=True).error(
                    f"error to infer current obs info: {e}"
                )
                return None
            if tbl is None:
                logger.error(
                    "unable to interfer current obs info: no symlink in data root."
                )
                return None
            tbl = guess_info_from_sources([resolve_symlink(p) for p in tbl["path"]])
            info = tbl.toltec_file.get_latest()
            logger.debug(f"resolved latest obs info:{pformat_yaml(info.model_dump())}")
            files = [info.filepath]
        elif re.match(r"^(.*/)?toltec.+\.nc", obs_spec):
            file = Path(obs_spec)
            logger.debug(f"resovled {file=}")
            if file.exists():
                files = [file]
            else:
                logger.error(f"resovled file does not exist: {file}")
                files = []
        else:
            info = dict_from_regex_match(
                r"^(?P<obsnum>\d+)"
                r"(?:-(?P<subobsnum>\d+)"
                r"(?:-(?P<scannum>\d+)(?:-(?P<roach>\d+))?)?)?",
                obs_spec,
                type_dispatcher={
                    "obsnum": int,
                    "subobsnum": int,
                    "scannum": int,
                    "roach": int,
                },
            )
            if info is not None:
                obsnum = info["obsnum"]
                subobsnum = info["subobsnum"]
                scannum = info["scannum"]
                roach = info["roach"]

                logger.debug(f"resolved {obsnum=} {subobsnum=} {scannum=} {roach=}")

                data_lmt_path = option.data_lmt_path
                p_interface = "toltec*" if roach is None else f"toltec{roach}"
                p_obsnum = f"_{obsnum:06d}"
                p_subobsnum = "_*" if subobsnum is None else f"_{subobsnum:03d}"
                p_scannum = "_*" if scannum is None else f"_{scannum:04d}"
                p = f"{p_interface}/{p_interface}{p_obsnum}{p_subobsnum}{p_scannum}_*.nc"
                glob_patterns = [
                    f"toltec/ics/{p}",
                    f"toltec/tcs/{p}",
                ]
                logger.debug(
                    f"search file pattern:\n{pformat_yaml(glob_patterns)}\nin {data_lmt_path=}"
                )
                files = list(
                    itertools.chain(*(data_lmt_path.glob(p) for p in glob_patterns))
                )
            else:
                logger.error(f"unknown {obs_spec=}")
                files = []
        if not files:
            logger.error(f"no files can be resolved from obs specifier {obs_spec}")
            return None
        files = [resolve_symlink(f) for f in files]
        logger.debug(
            f"resovled {len(files)} files from {obs_spec=}\n{pformat_yaml(files)}"
        )
        if unique:
            if len(files) > 1:
                raise ValueError(f"ambiguous files found from {obs_spec=}")
            return files[0]
        return files


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    LmtToltecPathOption.add_args_to_parser(parser)
    option = parser.parse_args()
    path_option = LmtToltecPathOption(option)

    files = path_option.get_raw_obs_files()

    if not files:
        raise ValueError(f"no file path found for obs_spec={option.obs_spec}")
    info = guess_info_from_sources(files).toltec_file.get_latest()
    # now get various extra
    info_extra = {
        "apt_filename": get_apt_filename(info),
        "obs_goal": get_obs_goal(path_option.lmt_fs, info),
    }
    output_lns = []
    output_fields = [
        "obsnum",
        "subobsnum",
        "scannum",
        "roach",
        "interface",
        "uid_obs",
        "uid_raw_obs",
    ]
    for field in output_fields:
        output_lns.append(f"tfu_{field}={getattr(info, field)}")

    for field, value in info_extra.items():
        output_lns.append(f"tfu_{field}={value if value is not None else ''}")

    print("\n".join(output_lns))
    # def _print_and_exit(v):
    #     print(v)
    #     sys.exit(0)
    #
    # if option.print_key == "apt":
    #     _print_and_exit(get_apt_filename(info))
    #
    # if option.print_key == "obs_goal":
    #     _print_and_exit(get_obs_goal(info))
    #
    # if option.print_key in source_info_fields:
    #     _print_and_exit(getattr(info, option.print))
    #

    # data_lmt_path = Path(option.data_lmt_path)
    # if not data_lmt_path.exists():
    #     raise ValueError(f"{data_lmt_path=} does not exist")
    # # make sure all paths exist
    # # raw data store
    # lmtfs = LmtFileStore(path=data_lmt_path)
    # toltecfs = lmtfs.toltec
    # if toltecfs is None:
    #     raise ValueError(f"no toltec data found in {data_lmt_path=}")
    # # dataprod store
    # # dataprod_path = Path(option.dataprod_path)
    # # if not dataprod_path.exists():
    # #     raise ValueError(f"{dataprod_path=} does not exist")
    # # dpfs = DataProdFileStore(path=dataprod_path)
    #
    # info = get_toltec_raw_obs_info(toltecfs, obsnum=option.obsnum)
