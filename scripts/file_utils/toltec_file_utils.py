from pathlib import Path
from tolteca_datamodels.lmt.filestore import LmtFileStore
from tollan.utils.log import logger
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


def get_toltec_raw_obs_info(totlecfs, obsnum=None):
    if obsnum is None:
        try:
            tbl = toltecfs.get_symlink_info_table()
        except Exception as e:
            logger.opt(exception=True).error(f"error to infer current obs info: {e}")
            return None
        if tbl is None:
            logger.error(
                "unable to interfer current obs info: no symlink in data root."
            )
            return None
        tbl = guess_info_from_sources(tbl["path"])
    else:
        obsnum = option.obsnum
        files = toltecfs.glob(f"toltec*_{obsnum:06d}_*.nc")
        tbl = guess_info_from_sources(files)
    info = tbl.toltec_file.get_latest()
    logger.debug(f"resolved info:{pformat_yaml(info.model_dump())}")
    return info


def get_lmt_info_tbl_from_toltec_info(lmtfs: LmtFileStore, info: SourceInfoModel):
    master = info.filepath.parent.parent.name
    logger.debug(f"inferred {master=} for {info=}")
    if master != "tcs":
        logger.error(f"no tel file for {master=} {info=}")
        return None
    glob_pattern = f"tel_*{info.obsnum:06d}_{info.subobsnum:02d}_{info.scannum:04d}.nc"
    files = list(lmtfs.tel_path.glob(glob_pattern))
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


def get_obs_goal(lmtfs, info: SourceInfoModel):
    tel_info_tbl: SourceInfoDataFrame = get_lmt_info_tbl_from_toltec_info(lmtfs, info)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obsnum", default=None, type=int)
    parser.add_argument("--data_lmt_path", default="/data_lmt")
    # parser.add_argument("--dataprod_path", default="/dataprod_toltec")
    parser.add_argument("--verbose", action="store_true")

    option = parser.parse_args()
    data_lmt_path = Path(option.data_lmt_path)
    if not data_lmt_path.exists():
        raise ValueError(f"{data_lmt_path=} does not exist")
    # make sure all paths exist
    # raw data store
    lmtfs = LmtFileStore(path=data_lmt_path)
    toltecfs = lmtfs.toltec
    if toltecfs is None:
        raise ValueError(f"no toltec data found in {data_lmt_path=}")
    # dataprod store
    # dataprod_path = Path(option.dataprod_path)
    # if not dataprod_path.exists():
    #     raise ValueError(f"{dataprod_path=} does not exist")
    # dpfs = DataProdFileStore(path=dataprod_path)

    info = get_toltec_raw_obs_info(toltecfs, obsnum=option.obsnum)
    # now get various extra
    info_extra = {
        "apt_filename": get_apt_filename(info),
        "obs_goal": get_obs_goal(lmtfs, info),
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
        output_lns.append(f"tfu_{field}={value}")

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
