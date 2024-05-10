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
    echo "Usage: $0 file [tolteca_kids_options]"
    exit 1
fi
obsnum=$1
obsnum_str=$(printf "%06d" ${obsnum})
for file in ${dataroot}/toltec/{ics,tcs}/toltec*/toltec*_${obsnum_str}_*_{vnasweep,targsweep,tune}.nc; do
    if [[ ! -f ${file} ]]; then
        echo "file does not exist: ${file}"
	continue
    fi
    bash ${scriptdir}/dispatch_tolteca_kids.sh ${file}
done
