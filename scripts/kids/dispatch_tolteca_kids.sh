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
obs_spec=$1
shift

echo "processing ${obs_spec}"
dispatch_py ${obs_spec} \
    --data_lmt_path ${dataroot} \
    --config ${scriptdir}/tolteca_config.d \
    --kids.output.path ${scratchdir} \
    --kids.sweep_check_plot.save_path ${scratchdir} \
    --kids.kids_find_plot.save_path ${scratchdir}  \
    --kids.tlaloc_output.enabled \
    --kids.tlaloc_output.path ${tlalocetcdir} \
    "$@"
