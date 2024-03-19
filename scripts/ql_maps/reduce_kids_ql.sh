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

qldir=${dataprodroot}/ql

if [[ ! $1 ]]; then
    echo "Usage: reduce_kids_ql.sh filepaths"
fi
filepaths="$@"
output_dir=${qldir}

echo "kids ql reduce for ${filepaths} ${output_dir}"

set -x
${pybindir}/python3 ${scriptdir}/reduce_kids_ql.py ${filepaths} \
    --output_dir ${output_dir} --log_level INFO \
    --search_paths ${dataroot}/toltec_clip{a,o}/reduced

set +x

