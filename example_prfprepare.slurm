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

# Make sure FS_LICENSE is defined in the container.
export SINGULARITYENV_FS_LICENSE=/flywheel/v0/BIDS/.freesurfer.txt

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${basedir}/BIDS/participants.tsv )
echo 

# Removed the --no-home directive because it was failing couldnot find run.py...
SINGULARITY_CMD="module load Singularity/3.5.3-GCC-8.3.0 && \
                 unset PYTHONPATH && \
                 singularity run --cleanenv --home /scratch/glerma \
                     -B ${basedir}/BIDS/derivatives/fmriprep_21.0.2:/flywheel/v0/input \
                     -B ${basedir}/BIDS/derivatives:/flywheel/v0/output  \
                     -B ${basedir}/BIDS:/flywheel/v0/BIDS  \
                     -B ${basedir}/config${subject}.json:/flywheel/v0/config.json
                      /scratch/glerma/containers/prfprepare_1.0.2.sif"

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
# FROM glerma@atlas-edr-login-02:/scratch/glerma/DATA/VWFA_FOV_HEB/BIDS:$
# Execute this. It is relevant to execute from this folder and not from Nifti or BIDS
# sbatch --array=1-$(( $( wc -l $basedir/BIDS/participants.tsv | cut -f1 -d' ' ) - 1 )) /dipc/glerma/soft/prfprepare/example_prfprepare.slurm

