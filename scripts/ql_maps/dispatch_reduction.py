#!/usr/bin/env python

from pathlib import Path
from tolteca.datamodels.toltec import BasicObsDataset
from tollan.utils import call_subprocess_with_live_output
import pty
import shlex
from file_utils_v1 import (
        get_dataset_of_latest_obsnum,
        get_dataset,
        get_obs_goal,
        )

scriptdir = Path(__file__).parent


def shell_run(cmd):
    import shlex
    import subprocess
    from io import TextIOWrapper

    def _handle_ln(ln):
        sys.stdout.write(ln)

    with subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            shell=True,
            ) as proc:
        reader = TextIOWrapper(proc.stdout, newline='')
        for ln in iter(
                reader.readline, b''):
            _handle_ln(ln)
            if proc.poll() is not None:
                sys.stderr.write('\n')
                break
        retcode = proc.returncode
        if retcode:
            # get any remaining message
            ln, _ = proc.communicate()
            _handle_ln(ln.decode())
            _handle_ln(
                f"The process exited with error code: {retcode}\n")
            return False
        return True


if __name__ == "__main__":
    from tollan.utils.log import init_log
    # init_log(level='DEBUG')
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('obsnum', nargs='?', default=None, type=int)
    parser.add_argument('--data_lmt_path', '-d', default='/data_lmt')

    option = parser.parse_args()
    data_lmt_path = Path(option.data_lmt_path)
    print(f"use {data_lmt_path=}")
    if option.obsnum is None:
        dataset = get_dataset_of_latest_obsnum(data_lmt_path=data_lmt_path)
        if dataset is None:
            print('no file found, abort')
            sys.exit(1)
        obsnum = dataset[-1]['obsnum']
    else:
        obsnum = option.obsnum
        dataset = get_dataset(data_lmt_path=data_lmt_path, obsnum=obsnum)
        if dataset is None:
            print(f'no file found for obsnum={obsnum}, abort')
            sys.exit(1)
    print(dataset)

    # diaptch based on type
    if dataset[0]['master_name'] == 'ics':
        print(f"no runs defined for ics data, nothing to do.")
        # print('run {reduce_all.sh}')
        # shell_run(f'./reduce_all_seq.sh {obsnum}')
        # pty.spawn(shlex.split(f'{scriptdir}/reduce_all_seq_new.sh {obsnum}'))
    elif dataset[0]['master_name'] == 'tcs':
        # cehck the tel obsgola for the obsgaol
        obs_goal = get_obs_goal(dataset)
        print(f"found {obs_goal=}")
        if obs_goal is None:
            print(f"no runs defined for {obs_goal=}, nothing to do.")
            # print('run general reduction')
            # pty.spawn(shlex.split(f'{scriptdir}/reduce_all_seq_new.sh {obsnum}'))
        else:
            # need to get all tune files reduced
            for filepath in dataset['source']:
                if filepath.endswith('tune.nc'):
                    pass
                    # pty.spawn(shlex.split(f'{scriptdir}/reduce.sh {filepath}'))
        if obs_goal == 'beammap' or obs_goal == 'azscan' or obs_goal == 'elscan':
            print('run beammap')
            pty.spawn(shlex.split(f'bash {scriptdir}/reduce_beammap.sh {obsnum}'))
        elif obs_goal in ['pointing', 'focus', 'astigmatism', 'm3offset', 'oof']:
            print('run pointing')
            pty.spawn(shlex.split(f'{scriptdir}/reduce_pointing.sh {obsnum}'))
        elif obs_goal == 'science':
            print('run science')
            pty.spawn(shlex.split(f'{scriptdir}/reduce_science.sh {obsnum}'))
        else:
            print(f'unknown obs goal: {obs_goal}, no action.')
            # print('run pointing for test')
            # pty.spawn(shlex.split(f'./reduce_pointing.sh {obsnum}'))
    else:
        print('unknown data, ignore')
