#!/bin/bash
#
#SBATCH -J prfprepare 
#SBATCH --time=01:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24000
#SBATCH --partition regular  # Queue names you can submit to
# Outputs ----------------------------------
#SBATCH -o /scratch/glerma/logs/%x-%A-%a.out
#SBATCH -e /scratch/glerma/logs/%x-%A-%a.err
# ------------------------------------------



subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${basedir}/BIDS/participants.tsv )
echo 

SINGULARITY_CMD="module load Singularity/3.5.3-GCC-8.3.0 && \
                 unset PYTHONPATH && \
                 singularity run --cleanenv --no-home \
                     -B ${basedir}/BIDS/derivatives/fmriprep_21.0.2:/flywheel/v0/input \
                     -B ${basedir}/BIDS/derivatives:/flywheel/v0/output  \
                     -B ${basedir}/BIDS:/flywheel/v0/BIDS  \
                     -B ${basedir}/config${subject}.json:/flywheel/v0/config.json
                      /scratch/glerma/containers/prfprepare_1.0.1.sif"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}, for subject ${subject}
echo Commandline: $SINGULARITY_CMD
eval $SINGULARITY_CMD
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID} $exitcode" >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode



# TO LAUNCH IT
# export STUDY=/scratch/glerma/DATA/projectName/Nifti
# cd $STUDY
# sbatch --array=1-$(( $( wc -l $STUDY/participants.tsv | cut -f1 -d' ' ) - 1 )) ../fmriprep.slurm

