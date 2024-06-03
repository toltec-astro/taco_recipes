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
# exec 1>>${logdir}/dispatch_reduction.log
# exec 2>&1

${pybindir}/python3 ${scriptdir}/dispatch_reduction.py \
    $@  \
    --data_lmt_path=${dataroot} >> ${logdir}/dispatch_reduction.log 2>&1
