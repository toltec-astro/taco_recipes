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
set -eo pipefail

function file_utils_py_call {
    ${pybindir}/python3 ${scriptdir}/file_utils_v1.py \
        $@ \
        --data_lmt_path=${dataroot}
}

commondir=$HOME/toltec_astro/run/tolteca/common
rcdir=$HOME/toltec_astro/run/tolteca/pointing

if [[ ! $1 ]]; then
    obsnum=$(file_utils_py_call obsnum)
    echo found latest obsnum ${obsnum}
else
    obsnum=$1
fi
scannum=$(file_utils_py_call scannum --obsnum ${obsnum})
echo found latest scannum ${scannum}

apt_filename=$(file_utils_py_call apt --obsnum ${obsnum})
echo use apt filename ${apt_filename}

echo "reduce pointing obsnum=${obsnum}"

obsnum_str=$(printf "%06d" ${obsnum})

tune_obsnum=$((${obsnum} -1))
tune_obsnum_str=$(printf "%06d" ${tune_obsnum})
echo "tune_obsnum=${tune_obsnum}"
#
# link files to input folder
set -x
tel_file=${dataroot}/tel/tel_toltec*_${obsnum_str}_*.nc

ln -sf ${tel_file} ${rcdir}/data/
ln -sf ${dataroot}/toltec/tcs/toltec*/toltec*_${obsnum_str}_*.nc ${rcdir}/data/
ln -sf ${dataroot}/toltec_clip{a,o}/reduced/toltec*_${obsnum_str}_*.txt ${rcdir}/data/
ln -sf ${dataroot}/toltec_clip{a,o}/reduced/toltec*_${tune_obsnum_str}_*.txt ${rcdir}/data/

set +e


# run match apt with current obs

# apt_in_file=${commondir}/apt_GW_2024_v4.ecsv #apt_GW_v8_with_fg_pg_loc_ori_flipped_flag.ecsv
apt_in_file=${commondir}/apt.ecsv
time ${pybindir}/python3 ${scriptdir}/make_matched_apt_fixed.py \
    --data_rootpath ${dataroot}\
    --apt_in_file ${apt_in_file} \
    --output_dir ${rcdir}/data \
    -- ${obsnum}

apt_matched_file=${rcdir}/data/apt_${obsnum}_matched.ecsv

set +x
ln -sf ${apt_matched_file} ${rcdir}/data/${apt_filename}

# run tolteca reduce
time $toltecaexec -d ${rcdir} -g -- reduce --jobkey reduced/${obsnum} \
    --inputs.0.select "obsnum == ${obsnum} & (scannum == ${scannum})" \
    --steps.0.path ~/toltec_astro/citlali/build/bin/citlali
# $toltecaexec -g -d ${rcdir} -- reduce --jobkey reduced/${obsnum} --inputs.0.select "(obsnum == ${obsnum}) & (scannum == ${scannum}) & (interface != \"toltec0\") & (interface != \"toltec6\")"
# & (interface != \"toltec4\") & (interface != \"toltec6\") " #& (interface != \"toltec2\") & (interface != \"toltec3\") " #& (interface != \"toltec1\") & (interface != \"toltec4\") & (interface != \"toltec6\")"

# run the pointing script
resultdir=${rcdir}/reduced/${obsnum}
redudir=$(${pybindir}/python3 ${scriptdir}/get_largest_redu_dir_for_obsnum.py $resultdir $obsnum)
if [[ $? != 0 ]]; then
    exit 0
fi
echo "run pointing reader in ${redudir}"
set -x
time ${pybindir}/python3 $scriptdir/pointing_reader_v1_5.py -c crpix -p ${redudir}/${obsnum}/raw --obsnum ${obsnum} -s -o ${redudir}/${obsnum}/raw

