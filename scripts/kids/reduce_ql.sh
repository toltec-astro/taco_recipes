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

qldir=${dataprodroot}/ql

if [[ ! $1 ]]; then
    echo "Usage: $0 obs_spec [obs_spec ...]"
fi
obs_specs="$@"
output_dir=${qldir}

echo "run kids ql reduce for ${obs_specs}"
echo "output_dir: ${output_dir}"
dispatch_py ${obs_specs} \
    --log_level DEBUG \
    --output_dir ${output_dir} \
    --search_paths \
    ${scratchdir} \
    ${dataroot}/toltec/reduced \
    ${dataroot}/toltec_reduced_taca \
    ${dataroot}/toltec_clip{a,o}/reduced \
    --data_lmt_path ${dataroot} \
    > ${logdir}/reduce_kids_ql.log

