#!/bin/bash
TARGET_USER="toltec"

if [ "$EUID" -eq 0 ]; then
    echo "Running as root. Switching to $TARGET_USER..."
    scriptdir=$(dirname "$(readlink -f "$0")")
    scriptpath=${scriptdir}/$(basename $0)
    su - "$TARGET_USER" -c "${scriptpath} $*"
    echo "Back to root."
    exit 0
fi

# Actual script logic goes here
echo "Running as user: $(whoami)"


source /home/toltec/toltec_astro/dotbashrc

file=${@: 1:1}
args_all=${@: 2:$#-1}
link=$(readlink ${file})
if [[ ! ${link} ]]; then
    link=${file}
fi

##### old stuff, to be removed
# filter out --output arg
# filter out -r/-p
args=()
runmode="reduce"
shift # start with args after ${file} 
while [[ "$#" > 0 ]] ; do
    if [[ ${1} == "--output" ]]; then
        outfile=${2}
        shift
    # args+=( ${@: 3:$#-2} )
    # break
    elif [[ ${1} == "--action" ]]; then
        action=${2}
        shift
    elif [[ ${1} == "-r" ]]; then
        runmode="reduce"
    elif [[ ${1} == "-p" ]]; then
        runmode="plot"
    elif [[ ${1} == "-f" ]]; then
        runmode="fg"
    else
        args+=( ${1} )
    fi
    shift
done
args=$(echo ${args[@]})
echo "file: ${file}"
echo "link: ${link}"
echo "outfile: ${outfile}"
echo "args: ${args}"
echo "action: ${action}"
################################

scriptdir=$(dirname "$(readlink -f "$0")")
echo "scriptdir: ${scriptdir}"

scratchdir=/data/data_toltec/reduced
if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"

# dispatch drive fit

dispatch_tlaloc_bin=${scriptdir}/../dispatch_tlaloc.sh

# resolve file name for remote exec
remote_exec_host=taca
# remote_exec_host=
if [[ ! ${remote_exec_host} ]]; then
    function run_tlaloc_dispatch {
        echo "run dispatch tlaloc bin locally"
        bash ${dispatch_tlaloc_bin} ${file} --etc_path ${HOME}/tlaloc/etc "$@"
    }
else
    function run_tlaloc_dispatch {
        echo "run dispatch tlaloc bin on ${remote_exec_host}"
	remote_file=${HOME}/toltec_astro/run/data_lmt/toltec/${link}
	remote_tlaloc_etc=${HOME}/toltec_astro/run/tlaloc_etc
        ssh -oStrictHostKeyChecking=accept-new -t ${remote_exec_host} bash ${dispatch_tlaloc_bin} ${remote_file} --etc_path ${remote_tlaloc_etc} "$@"
    }
fi 
if [[ ${action} =~ "drivefit_reduce" ]]; then
    echo "run drive fit reduce"
    set -x
    run_tlaloc_dispatch --action ${action} 2>&1
    set +x
    exit 0
fi
if [[ ${action} =~ "drivefit_commit" ]]; then
    echo "run drive fit commit"
    set -x
    run_tlaloc_dispatch --action ${action} 2>&1
    set +x
    exit 0
fi

upload_tones=1
if [[ ${link} == *"vnasweep"* ]]; then
    type="vna"
elif [[ ${link} == *"targsweep"* ]]; then
    type="targ"
elif [[ ${link} == *"tune"* ]]; then
    type="targ"
    upload_tones=0
elif [[ ${link} == *"timestream"* ]]; then
    type="timestream"
else
    type="timestream"
    echo "unrecognized type ${file}, exit."
    # exit 1
fi
echo "running $type reduction, upload_tones=${upload_tones}"
echo "additional output to: ${scratchdir}"

kidscppdir="${HOME}/toltec_astro/kidscpp"
# kidspydir="${HOME}/zma_deprecated/kids_master/scripts"
# pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
# pyexec_v1="${HOME}/toltec_astro/extern/pyenv/versions/tolteca_v1/bin/python3"
# pyexec=${HOME}/.pyenv/versions/tolteca_v1/bin/python3
# pyexec_v1=${HOME}/.pyenv/versions/tolteca_v1/bin/python3
# finder_thresh=10  # TODO need a better way to handle this
# fitter_Qr=13000  # TODO need a better way to handle this

filename=$(basename ${link})
echo ${filename}
if [[ "${filename}" =~ ^toltec([0-9][0-9]?)_.+ ]]; then
  nw=${BASH_REMATCH[1]}
else
  nw=0
fi
echo "nw: ${nw}"
if (( ${nw} >= 7 )); then
    # 1.4 and 2.0mm
    finder_args=( \
        --fitter_weight_window_Qr 6500 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --finder_threshold 3 --finder_stats_clip_sigma 2 --fitter_lim_gain_min 0)
    fitter_args=( \
        --fitter_weight_window_Qr 6500 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --fitter_auto_global_shift)
elif (( ${nw} <= 6 )); then
    finder_args=( \
        --fitter_weight_window_Qr 10000 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --finder_threshold 3 --finder_stats_clip_sigma 2 --fitter_lim_gain_min 0)
    fitter_args=( \
        --fitter_weight_window_Qr 10000 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --fitter_auto_global_shift)
fi
echo finder_args: ${finder_args[@]}

if [[ ${type} == "vna" ]]; then
    echo "do ${type} ${runmode}"
    reportfile=$(basename ${link})
    reportfile=${reportfile%.*}.txt
    echo "reportfile: ${reportfile}"
    reportfile="${scratchdir}/${reportfile}"
    if [[ ${runmode} == "reduce" ]]; then
        ${kidscppdir}/build/bin/kids \
            ${finder_args[@]} \
            --output_d21 ${scratchdir}/'{stem}_d21.nc' \
            --output_processed ${scratchdir}/'{stem}_processed.nc' \
            --output "${reportfile}" \
                "${file}" ${args}
        cp ${reportfile} ${reportfile}.kidscpp_v0
    elif [[ ${runmode} == "plot" ]]; then
	echo "not implemented!"
	exit 1
    elif [[ ${runmode} == "fg" ]]; then
	echo "not implemented!"
	exit 1
    fi
elif [[ ${type} == "targ" ]]; then
    echo "do ${type} ${runmode}"
    reportfile=$(basename ${link})
    reportfile=${reportfile%.*}.txt
    echo "reportfile: ${reportfile}"
    reportfile="${scratchdir}/${reportfile}"
    if [[ ${runmode} == "reduce" ]]; then
        # this is legacy kids reduce tune.txt
        ${kidscppdir}/build/bin/kids \
           ${fitter_args[@]} \
           --output_processed ${scratchdir}/'{stem}_processed.nc' \
           --output "${reportfile}"  "${file}" ${args}
        cp ${reportfile} ${reportfile}.kidscpp_v0
    elif [[ ${runmode} == "plot" ]]; then
        ${pyexec} ${kidspydir}/kidsvis.py ${file} --fitreport "${reportfile}" --use_derotate ${args} --grid 8 8 &
    elif [[ ${runmode} == "fg" ]]; then
        ${pyexec} ${kidspydir}/kidsvis.py ${file} --fitreport "${reportfile}" --use_derotate ${args} --grid 8 8 
    fi
elif [[ ${type} == "timestream" ]]; then
    echo "do ${type} ${runmode}"
    if [[ ${runmode} == "reduce" ]]; then
        o=$(basename ${file})
        o=${scratchdir}/${o%.*}_processed.nc
        if [ -f "${o}" ] ; then
            echo rm ${o}
            rm "${o}"
        fi
        ${kidscppdir}/build/bin/kids \
            --solver_fitreportdir ${scratchdir} \
            --output "${scratchdir}/{stem}_processed.nc" --solver_chunk_size 500000 --solver_extra_output ${file} ${args}
    elif [[ ${runmode} == "plot" ]]; then
        ${pyexec} ${kidspydir}/timestream.py --fitreportdir ${scratchdir} ${file} ${args} &
    elif [[ ${runmode} == "fg" ]]; then
        ${pyexec} ${kidspydir}/timestream.py --fitreportdir ${scratchdir} ${file} ${args} --grid 1 1
    fi
fi

# dispatch_tlaloc_bin=${scriptdir}/../dispatch_tlaloc.sh
# bash ${dispatch_tlaloc_bin} ${file} --etc_path ${HOME}/tlaloc/etc --action ${action} 2>&1
run_tlaloc_dispatch --action ${action} 2>&1
