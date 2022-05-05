# dlinhardt/prfprepare

This repo is designed for a seamless experience of pRF mapping and is closing the gap between preprocessing (using e.g. fmriprep) and the pRF analysis (using github.com/vistalab/PRFmodel).

#### INSTALL
You can pull the latest docker: `docker pull davidlinhardt/prfprepare:1.0.3`  
Or as singularity image:        `singularity build prfprepare_1.0.3.sif docker://davidlinhardt/prfprepare:1.0.3`

#### WORKFLOW
The docker is built up in three stages:
  1) build the stimuli from stimulus image and vistadisp log files
  2) create a mask in fsnative space containing all specified areas (benson, wang atlas) and output corresponding information, apply this mask and convert the preprocessed surface bold files to 2D NIFIT2 files as preparation for prfanalyze
  4) link the correct stimulus appertures to the respective bold files by creating events.tsv

#### BEFORE
For running prfprepare you need to provide the following folder structure:  

<BASE_DIR>/BIDS has to contain:  
 -) the preprocessed data in fsnative space and the corresponding freesurfer segmenataion (fmriprep output in analysis-XX) 
 -) the log file from vistadisp   
 -) unprocessed subject data in BIDS-compatible format (e.g. from heudiconv)  
 
 <BASE_DIR>/BIDS/
![image](https://user-images.githubusercontent.com/41369769/166925490-30f03cb6-9baf-42c3-bdf1-e9c89c56d3dd.png)

#### RUNNING prfprepare
We provide an example rundocker.sh script with all necessary bindings and an example_config.json file in this repository.  
example_config.json, * mandatory  
{  
    "subject"                 : "001",        #* define the subject number here  
    "session"                 : "all",        #  define sessions as "[001,002]" or "all", if not specified it will take all  
    "fs_annot"                : "custom.zip", #  WIP  
    "fs_annot_ids"            : "",           #  WIP  
    "etcorrection"            : false,        #  we can perform eyetracker correction, if not speciefied false  
    "force"                   : false,        #  fore redo all stepss, if not specified false  
    "custom_output_name"      : "",           #  we can write a link with the given name to the output folder, if not speciefied false  
    "fmriprep_legacy_layout"  : false,        #  true for fmriprep versions <21.0.0   
    "forceParams"             : "",           #  instead of run-specific params file use the given one, WIP, see below  
    "use_numImages"           : true,         #  use number numImges or seqTimeing to create the stimulus, see below  
    "verbose"                 : true,         #  verbose output  
    "config": {                         #*  this part will define if we start a new output direcotory or not  
        "average_runs"        : true,         #  average runs of the same task, if not speciefied false  
        "output_only_average" : true,         #  output only the average, if not speciefied false  
        "rois"                : "all",        #  define the ROIs to output as "[V1,V2,...]" or "all", if not speciefied it will take all  
        "atlases"             : "all",        #  define the atlases to output as "wang" or "all", if not speciefied it will take all (wang,benson)  
        "fmriprep_analysis"   : "01"          #  define the fmriprep analysis folder number, if not specified ie assumes "01"  
 	}  
}  



