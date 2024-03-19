#!/usr/bin/env python

from pathlib import Path
from tolteca.datamodels.toltec import BasicObsDataset
from tollan.utils import call_subprocess_with_live_output
import pty
import shlex

scriptdir = Path(__file__).parent


def get_dataset_of_latest_obsnum(data_lmt_path='/data_lmt'):
    data_lmt_path = Path(data_lmt_path)
    data_toltec_path = data_lmt_path.joinpath('toltec')
    links = (
            list(data_toltec_path.glob('toltec[0-9].nc'))
            + list(data_toltec_path.glob('toltec1[0-2].nc'))
            + list(data_toltec_path.glob('hwpr.nc')))
    if not links:
        return None
    dataset = BasicObsDataset.from_files([link.resolve() for link in links])
    dataset.sort(['scannum', 'ut'])
    obsnum = dataset[-1]['obsnum']
    print(dataset.index_table)
    # return dataset[dataset['obsnum'] == obsnum]
    return BasicObsDataset(dataset.index_table[dataset['obsnum'] == obsnum])


def get_dataset(data_lmt_path, obsnum):
    links = (
        list(data_lmt_path.glob(f"toltec/tcs/toltec[0-9]*/toltec[0-9]*_{obsnum:06d}_*.nc"))
        + list(data_lmt_path.glob(f"toltec/ics/toltec[0-9]*/toltec[0-9]*_{obsnum:06d}_*.nc"))
        + list(data_lmt_path.glob(f"tel/tel_toltec_*_{obsnum:06d}_*.nc"))
        + list(data_lmt_path.glob(f"toltec/ics/wyatt/wyatt_*_{obsnum:06d}_*.nc"))
        )
    if not links:
        return None
    dataset = BasicObsDataset.from_files([link.resolve() for link in links])
    dataset.sort(['interface', "scannum"])
    return dataset


def get_obsnum(dataset):
    return dataset[-1]['obsnum']

def get_subobsnum(dataset):
    return dataset[-1]['subobsnum']

def get_scannum(dataset):
    return max(s for s in dataset.index_table['scannum'])

def get_obs_goal(dataset):
    if dataset[0]['master_name'] != 'tcs':
        return None
    try:
        bod_tel = dataset[dataset['interface'] == 'lmt'].bod_list[0].open()
        obs_goal = bod_tel.meta['obs_goal']
        return obs_goal
    except Exception:
        return None


def get_apt_filename(dataset):
    f = Path(dataset[-1]['source']).resolve()
    stem = f.stem.split("_", 1)[-1]
    return f"apt_{stem}.ecsv"


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obsnum', default=None, type=int)
    parser.add_argument('--data_lmt_path', '-d', default='/data_lmt')
    parser.add_argument('print', choices=['apt', 'obsnum', 'scannum'])

    option = parser.parse_args()
    data_lmt_path = Path(option.data_lmt_path)
    if option.obsnum is None:
        dataset = get_dataset_of_latest_obsnum(data_lmt_path=data_lmt_path)
        if dataset is None:
            sys.exit(1)
    else:
        obsnum = option.obsnum
        dataset = get_dataset(data_lmt_path=data_lmt_path, obsnum=obsnum)
        if dataset is None:
            sys.exit(1)

    def _print_and_exit(v):
        print(v)
        sys.exit(0)

    if option.print == 'apt':
        _print_and_exit(get_apt_filename(dataset))

    if option.print == 'obsnum':
        _print_and_exit(get_obsnum(dataset))

    if option.print == 'scannum':
        _print_and_exit(get_scannum(dataset))
