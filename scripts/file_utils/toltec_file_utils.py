from __future__ import annotations
from pathlib import Path
from tolteca_datamodels.lmt.filestore import LmtFileStore
from tolteca_datamodels.toltec.filestore import ToltecFileStore
from typing import ClassVar
import pandas as pd
import itertools
from tollan.utils.log import logger
import re
from types import SimpleNamespace
import argparse
from tollan.utils.general import dict_from_regex_match, resolve_symlink
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


def get_lmt_info_table_from_toltec_info(
    lmt_fs: LmtFileStore,
    info: SourceInfoModel,
    query=None,
) -> SourceInfoDataFrame:
    """Return LMT data files for toltec info."""
    master = info.filepath.parent.parent.name
    if master not in ToltecFileStore.masters:
        logger.error(f"unable to infer toltec master from {info=}")
        return None
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
    logger.debug(f"resolved lmt info table:\n{tbl.toltec_file.pformat()}")
    if query is not None:
        return tbl.query(query)
    return tbl


def get_apt_filename(info: SourceInfoModel):
    # TODO: make rule for apt file.
    stem = info.source.path.stem.split("_", 1)[-1]
    return f"apt_{stem}.ecsv"


def get_obs_goal(lmt_fs, info: SourceInfoModel):
    if lmt_fs is None:
        logger.error("no valid data_lmt path.")
        return None
    tel_info_tbl = get_lmt_info_table_from_toltec_info(
        lmt_fs, info, query="interface == 'tel_toltec'"
    )
    if len(tel_info_tbl) == 0:
        logger.error("no tel data found.")
        return None
    if len(tel_info_tbl) > 1:
        raise ValueError("ambiguous tel data found.")
    tel_info = tel_info_tbl.toltec_file.to_info_list()[0]
    tel_filepath = tel_info.source.path
    tel_nc = netCDF4.Dataset(tel_filepath)
    obs_goal = ncstr(tel_nc.variables["Header.Dcs.ObsGoal"]).lower()
    tel_nc.close()
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
        self._obs_spec = option.obs_spec

    @classmethod
    def add_args_to_parser(
        cls,
        parser: argparse.ArgumentParser,
        defaults=None,
        obs_spec_required=False,
        obs_spec_multi=False,
    ):
        defaults = defaults or {}

        def _get_path_arg_kw(key):
            if key in defaults:
                return {"default": defaults[key]}
            return {}

        cls.add_data_lmt_path_argument(parser, **_get_path_arg_kw("data_lmt_path"))
        cls.add_dataprod_path_argument(parser, **_get_path_arg_kw("dataprod_path"))
        cls.add_tlaloc_etc_path_argument(parser, **_get_path_arg_kw("tlaloc_etc_path"))
        cls.add_obs_spec_argument(
            parser,
            required=obs_spec_required,
            multi=obs_spec_multi,
        )

    _default_paths: ClassVar = {
        "data_lmt_path": "/data_lmt",
        "dataprod_path": "/dataprod_toltec",
        "tlaloc_etc_path": "/tlaloc_etc",
    }

    @classmethod
    def add_data_lmt_path_argument(
        cls,
        parser: argparse.ArgumentParser,
        default=_default_paths["data_lmt_path"],
    ):
        parser.add_argument(
            "--data_lmt_path",
            type=Path,
            default=default,
            help="data_lmt path.",
        )

    @classmethod
    def add_dataprod_path_argument(
        cls,
        parser: argparse.ArgumentParser,
        default=_default_paths["dataprod_path"],
    ):
        parser.add_argument(
            "--dataprod_path",
            type=Path,
            default=default,
            help="data product path.",
        )

    @classmethod
    def add_tlaloc_etc_path_argument(
        cls,
        parser: argparse.ArgumentParser,
        default=_default_paths["tlaloc_etc_path"],
    ):
        parser.add_argument(
            "--tlaloc_etc_path",
            type=Path,
            default=default,
            help="tlaloc etc path.",
        )

    @classmethod
    def add_obs_spec_argument(
        cls,
        parser: argparse.ArgumentParser,
        required=False,
        multi=False,
    ):
        if multi and required:
            kw = {
                "nargs": "+",
            }
        elif multi:
            kw = {
                "nargs": "*",
                "default": None,
            }
        elif required:
            kw = {}
        else:
            kw = {
                "nargs": "?",
                "default": None,
            }
        parser.add_argument(
            "obs_spec",
            help="specifier of obs data.",
            **kw,
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

    @property
    def obs_spec(self):
        """The obs spec."""
        return self._obs_spec

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
        return getattr(option, "dataprod_path", None)

    @classmethod
    def _get_tlaloc_etc_file_store(cls, option):
        # TODO: add tlaloc etc file store handling
        return getattr(option, "tlaloc_etc_path", None)

    def get_raw_obs_info_table(
        self,
        raise_on_multiple=False,
        raise_on_empty=False,
    ):
        obs_spec = self.obs_spec
        if not isinstance(obs_spec, list):
            obs_spec = [obs_spec]
        result = []
        lmt_fs = self.lmt_fs
        toltec_fs = self.toltec_fs
        for s in obs_spec:
            r = self._get_raw_obs_info_table(lmt_fs, toltec_fs, s)
            if r is None:
                continue
            if isinstance(r, list):
                r = guess_info_from_sources([resolve_symlink(f) for f in r])
            result.append(r)
        result = pd.concat(result, ignore_index=True) if result else None
        n_files = 0 if result is None else len(result)
        if n_files > 0:
            logger.debug(
                f"resovled {n_files} files from {obs_spec=}\n{result.toltec_file.pformat()}"
            )
        if raise_on_multiple and n_files > 1:
            raise ValueError(
                f"ambiguous files found for {obs_spec=}:\n{result.toltec_file.pformat()}",
            )
        if raise_on_empty and n_files == 0:
            raise ValueError(f"no files found for {obs_spec=}")
        return result

    @staticmethod
    def _replace_parent_path(path, parent_path, ensure_exist=False):
        path = Path(path)
        parent_path = Path(parent_path)
        parent_name = parent_path.name
        for pp in path.parents:
            if pp.name == parent_name:
                subpath = path.relative_to(pp)
                break
        else:
            return None
        path = parent_path.joinpath(subpath)
        if ensure_exist and not path.exists():
            return None
        return path

    @classmethod
    def _get_raw_obs_info_table(cls, lmt_fs, toltec_fs, obs_spec):
        logger.debug(f"resovle {obs_spec=}")
        if obs_spec is None:
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
            logger.debug(f"toltec symlink info in {toltec_fs.path}:\n{tbl}")
            tbl = guess_info_from_sources([resolve_symlink(p) for p in tbl["path"]])
            # get latest raw obs group
            tbl_latest = tbl.toltec_file.get_raw_obs_latest()
            logger.debug(
                f"resolved latest raw obs info from {obs_spec=}:\n"
                f"{tbl_latest.toltec_file.pformat()}"
            )
            return tbl_latest
        if re.match(r"^(.*/)?toltec.+\.nc", obs_spec):
            file = Path(obs_spec)
            if file.exists():
                logger.debug(f"resovled {file=} from {obs_spec=}")
                return [file]
            # try to locate the file under the provided root
            data_lmt_path = lmt_fs.path
            logger.debug(
                f"search file with matched subpath for {file} in {data_lmt_path}",
            )
            file = cls._replace_parent_path(file, data_lmt_path, ensure_exist=True)
            if file is None:
                logger.error(f"no matched file found for {file} in {data_lmt_path}")
                return None
            logger.debug(f"resovled {file=} from {obs_spec=}")
            return [file]
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
        if info is None:
            logger.error(f"unable to resolve {obs_spec=} by regex match")
            return None
        obsnum = info["obsnum"]
        subobsnum = info["subobsnum"]
        scannum = info["scannum"]
        roach = info["roach"]
        logger.debug(
            f"resolved {obsnum=} {subobsnum=} {scannum=} {roach=} "
            f"from {obs_spec=} by regex match",
        )
        data_lmt_path = lmt_fs.path
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
            f"search file patterns in {data_lmt_path=}:\n"
            f"{pformat_yaml(glob_patterns)}",
        )
        files = list(itertools.chain(*(data_lmt_path.glob(p) for p in glob_patterns)))
        if not files:
            logger.error(f"no files found for {obs_spec=} in {data_lmt_path=}")
            return None
        return files


def to_bash_source(info: SourceInfoModel, extra=None, variable_prefix="tfu_"):
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

    def _make_var_name(field):
        return f"{variable_prefix}{field}"

    def _make_var(field, value):
        return f"{_make_var_name(field)}={value}"

    for field in output_fields:
        output_lns.append(_make_var(field, getattr(info, field)))

    extra = extra or None
    for field, value in extra.items():
        output_lns.append(_make_var(field, value if value is not None else ""))
    return "\n".join(output_lns)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    LmtToltecPathOption.add_args_to_parser(parser)
    option = parser.parse_args()
    path_option = LmtToltecPathOption(option)

    tbl = path_option.get_raw_obs_info_table(raise_on_empty=True)
    # get the latest item from the list
    info = tbl.toltec_file.get_info_latest()
    info_extra = {
        "apt_filename": get_apt_filename(info),
        "obs_goal": get_obs_goal(path_option.lmt_fs, info),
    }
    print(to_bash_source(info, extra=info_extra))
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
