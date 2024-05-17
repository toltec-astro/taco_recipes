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
    echo "Usage: $0 obs_spec [tolteca_kids_options]"
    exit 1
fi
dispatch_py \
    --config ${scriptdir}/tolteca_config.d \
    --data_lmt_path ${dataroot} \
    --dataprod_path ${scratchdir} \
    --tlaloc_etc_path ${tlalocetcdir} \
    "$@"
