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
file=$1
shift
outputdir=${dataroot}/toltec/reduced
if [[ ! -d ${outputdir} ]]; then
    outputdir=${scratchdir}
fi


echo "processing ${file}"
set -x
${pybindir}/python3 dispatch_tolteca_kids.py ${file} \
        --data_lmt_root ${dataroot} \
        --config ${scriptdir}/tolteca_config.d \
    --kids.sweep_check_plot.save_rootpath ${scratchdir} \
    --kids.kids_find_plot.save_rootpath ${scratchdir}  \
    --kids.output.subdir_fmt null \
    --kids.output.path ${outputdir} \
    --kids.tlaloc_output.enabled \
    --kids.tlaloc_output.path ${tlalocetcdir} \
    "$@"
set +x
