#!/bin/bash

## env settings

shenv_name="tolteca_v2"
pyenv_name="tolteca_v1"

## load env (do not edit)
scriptdir=$(dirname "$(readlink -f "$0")")
function _source() { local s="$1" ; shift ; source "$s" ; }
_source ${scriptdir}/../00_shenv.${shenv_name} ${pyenv_name}
print_env || { echo 'unable to load script env, exit.' ; exit 1; }

## actual script starts form here
# set -eo pipefail

function file_utils_py_call {
    ${pybindir}/python3 ${scriptdir}/file_utils_v1.py \
        $@ \
        --data_lmt_path=${dataroot}
}


if [[ ! $1 ]]; then
    obsnum=$(file_utils_py_call obsnum)
    echo found latest obsnum ${obsnum}
else
    obsnum=$1
fi

if [[ ! $2 ]]; then
    obsnum_current=$(file_utils_py_call obsnum)
    echo "DriveFit: found latest obsnum ${obsnum_current}"
else
    obsnum_current=${obsnum}
fi

if [[ ! $3 ]]; then
    nws=($(seq 0 12))
else
    nws=($3)
fi

perc=50

echo "DriveFit: commit results obsnum=${obsnum} to obsnum_current=${obsnum_current} perc=${perc}"
echo "running networks ${nws[@]}"


bin=${scriptdir}/get_ampcor_from_adrv.py
bin_lut_interp=${scriptdir}/add_lut_interp.py
bin_targ_amps=${scriptdir}/get_amps_for_freqs.py

obsnum_str=$(printf "%06d" ${obsnum})
obsnum_str_current=$(printf "%06d" ${obsnum_current})

for i in ${nws[@]}; do
    adrv_file=${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.csv
    adrv_log=${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.log
    if ! [ -f ${adrv_file} ]; then
        echo "skip nw=${i}, no adrv.csv file found."
        continue
    fi

    set -x
    ${pybindir}/python ${bin} -p ${perc} -- ${adrv_file} > ${adrv_log}

    if (( i < 7 )); then
        reduced=${dataroot}/toltec_clipa/reduced
    else
        reduced=${dataroot}/toltec_clipo/reduced
    fi
    ${pybindir}/python ${bin_lut_interp} \
       ${dataroot}/toltec/?cs/toltec${i}/toltec${i}_${obsnum_str}_001*_targsweep.nc \
       ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.p${perc}.txt \
       ${reduced}/toltec${i}_${obsnum_str_current}_*_targfreqs.dat \

    cp ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.p${perc}.lut.txt ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_default_targ_amps.dat
    set +x
    if (( i <= 6 )); then
        dest=clipa
    else
        dest=clipo
    fi
    scp ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_default_targ_amps.dat clipa:/home/toltec/tlaloc/etc/toltec${i}/
    echo "~~~~~~~ DriveFit result commited to dest=${dest} nw=${i}"
done
