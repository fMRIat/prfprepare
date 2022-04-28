#!/bin/bash
#
#SBATCH -J fmriprep
#SBATCH --time=48:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24000
#SBATCH --partition long  # Queue names you can submit to
# Outputs ----------------------------------
#SBATCH -o /scratch/glerma/logs/%x-%A-%a.out
#SBATCH -e /scratch/glerma/logs/%x-%A-%a.err
# ------------------------------------------

BIDS_DIR="$STUDY"
DERIVS_DIR="derivatives/fmriprep_21.0.2"
export HOMES=/scratch/glerma
LOCAL_FREESURFER_DIR="$STUDY/data/derivatives/freesurfer-7.2.0"

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME=$HOMES/.cache/templateflow
FMRIPREP_HOST_CACHE=$HOMES/.cache/fmriprep
FMRIPREP_WORK_DIR=$HOMES/.work/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}
mkdir -p ${FMRIPREP_WORK_DIR}

# Prepare derivatives folder
mkdir -p ${BIDS_DIR}/${DERIVS_DIR}

# This trick will help you reuse freesurfer results across pipelines and fMRIPrep versions
# mkdir -p ${BIDS_DIR}/derivatives/freesurfer-6.0.1
# if [ ! -d ${BIDS_DIR}/${DERIVS_DIR}/freesurfer ]; then
#         ln -s ${BIDS_DIR}/derivatives/freesurfer-6.0.1
#         ${BIDS_DIR}/${DERIVS_DIR}/freesurfer
#         fi

# Make sure FS_LICENSE is defined in the container.
export SINGULARITYENV_FS_LICENSE=$BIDS_DIR/.freesurfer.txt

# Designate a templateflow bind-mount point
export SINGULARITYENV_TEMPLATEFLOW_HOME="/templateflow"
# SINGULARITY_CMD="unset PYTHONPATH && singularity run --cleanenv --home /scratch/glerma \
#                  -B /scratch/glerma:/scratch/glerma \
SINGULARITY_CMD="unset PYTHONPATH && singularity run --cleanenv --no-home \
                 -B $BIDS_DIR:/data \
                 -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} \
                 -B ${FMRIPREP_HOST_CACHE}:/work \
                 /scratch/glerma/containers/fmriprep_21.0.2.sif"

                 # If you already have FS run, add this line to find it
                 # -B ${LOCAL_FREESURFER_DIR}:/fsdir \
# Remove IsRunning files from FreeSurfer
# find ${LOCAL_FREESURFER_DIR}/sub-$subject/ -name "*IsRunning*" -type f -delete

# Parse the participants.tsv file and extract one subject ID from the
# line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )


# Compose the command line
cmd="module load Singularity/3.5.3-GCC-8.3.0 &&  \
     ${SINGULARITY_CMD} \
     /data \
     /data/${DERIVS_DIR} \
     participant --participant-label $subject \
     -w /work/ -vv --omp-nthreads 8 --nthreads 12 --mem 16G \
     --output-spaces fsnative fsaverage \
     --use-aroma"
# Add these two lines if you had freesurfer run already
# \
#      --fs-subjects-dir /fsdir"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}, for subject ${subject}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID} $exitcode" >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode



# TO LAUNCH IT
# export STUDY=/scratch/glerma/DATA/projectName/Nifti
# cd $STUDY
# sbatch --array=1-$(( $( wc -l $STUDY/participants.tsv | cut -f1 -d' ' ) - 1 )) ../fmriprep.slurm

