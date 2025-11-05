#!/bin/bash
if [[ ! $1 ]]; then
    echo "$0 <obsspec>"
    exit 1
fi
obsspec=$1
bash dispatch_tolteca_kids.sh ${obsspec} \
    --log_level DEBUG \
    --kids.output.dump_context false \
    --kids.sweep_check.noise_psd true \
    --kids.sweep_check_plot.enabled true \
    --kids.sweep_check_plot.save true \
    --kids.kids_find.enabled false \
    --kids.kids_find_plot.enabled false \
    --kids.tlaloc_output.enabled false \
    --kids.output.enabled false


