#!/bin/bash

## env settings

shenv_name="tolteca_v2"
pyenv_name="tolteca_v2"

## load env (do not edit)
scriptdir=$(dirname "$(readlink -f "$0")")
function _source() { local s="$1" ; shift ; source "$s" ; }
_source ${scriptdir}/../00_shenv.${shenv_name} ${pyenv_name}
print_env || { echo 'unable to load script env, exit.' ; exit 1; }

## actual script starts form here


if [[ ! $1 ]]; then
    obsnum=$(${pybindir}/python3 ${scriptdir}/../utils/get_latest_obsnum.py)
    echo "DriveFit: found latest obsnum ${obsnum}"
else
    obsnum=$1
fi

echo "DriveFit: reduce obsnum=${obsnum}"

# obsnum_str=$(printf "%06d" ${obsnum})

# config_file=${scriptdir}/kid_phase_fit/config_20230503.yaml
config_file=${scriptdir}/kid_phase_fit/config_20240316.yaml
if [[ ! $2 ]]; then
    nws=($(seq 0 12))
else
    nws=($2)
fi
echo "running networks ${nws[@]}"

for nw in ${nws[@]}; do
    ${pybindir}/python3 ${scriptdir}/kid_phase_fit/kid_phase_fit.py \
        $config_file --network ${nw} --obsnum ${obsnum}
done

exit 0
