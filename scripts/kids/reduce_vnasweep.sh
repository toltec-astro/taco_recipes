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

if [[ ! -f $1 ]]; then
    echo "vnasweep file does not exist: ${1}"
    exit 1
fi
file=$1

echo "VnaSweep: reduce file ${file}"

exit 0
