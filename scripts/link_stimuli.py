#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:28:36 2021

@author: dlinhardt
"""

from nilearn import image
from nilearn.surface import vol_to_surf
import nibabel as nib
from os import path
import bids
import numpy as np
from shutil import copy2 as copy
import neuropythy as ny
import argparse
from glob import glob

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# parser
parser = argparse.ArgumentParser(description='parser for script converting mrVista stimulus files to nii')

parser.add_argument('sub',         type=str, help='subject name')
parser.add_argument('bids_in_dir', type=str, help='input directory before fmriprep for BIDS layout')
parser.add_argument('--etcorr',    type=str, help='perform an eyetracker correction [default: False]', default='False')
parser.add_argument('--areas',     type=str, help='which atlas to use for the region comparison [default: benson]', default='[V1]')
parser.add_argument('--force',     type=str, help='force a new run [default: False]', default='False')

args = parser.parse_args()

etcorr = str2bool(args.etcorr)
force  = str2bool(args.force)

# base paths
inP  = '/flywheel/v0/input'
outP = '/flywheel/v0/output/BIDS'

# get the bids layout fur given subject
layout = bids.BIDSLayout(args.bids_in_dir)

subs = layout.get(return_type='id', target='subject')

if args.sub in subs:
    sub = args.sub
else:
    exit(3)
    
# extract the ROIs
rois = args.areas.split(']')[0].split('[')[-1].split(',')

# go for all given ROIs
for roi in rois:
                
    sess = layout.get(subject=sub, return_type='id', target='session')  
    
    for sesI,ses in enumerate(sess):
    
        tasks = layout.get(subject=sub, session=ses, return_type='id', target='task')  
    
        for task in tasks:
            
            apertures = np.array(glob(path.join(outP, 'stimuli', 'task-*.nii.gz')))
            try:
                stimName  = apertures[[f'task-{task[:3]}' in ap for ap in apertures]].item()
            except:
                continue
            
            
            runs = layout.get(subject=sub, session=ses, task=task, return_type='id', target='run')
            
            for run in runs:
                
                # create events.tsv without ET
                newTSV = path.join(outP, f'sub-{sub}', f'ses-{ses}', 'func', 
                                   f'sub-{sub}_ses-{ses}_task-{task}-surf-{roi}_events.tsv')
                niiPa = newTSV.replace('events.tsv', f'run-{run}_desc-preproc_bold.nii.gz')
                
                if not path.exists(newTSV) or force:
                    with open(newTSV, 'w') as file:
                        
                        nii = nib.load(stimName)
                        TR  = nii.header['pixdim'][4]
                        nT  = nii.shape[3]
                        
                        # fill the events file
                        file.writelines('onset\tduration\tstim_file\tstim_file_index\n')
                        
                        for i in range(nT):
                            file.write(f'{i*TR:.3f}\t{TR:.3f}\t{stimName.split("/")[-1]}\t{i+1}\n')
                
                    
                
                # create events.tsv for ET corr
                if etcorr:
                    newTSV = path.join(outP, f'sub-{sub}', f'ses-{ses}', 'func', 
                                       f'sub-{sub}_ses-{ses}_task-{task}-surf-etcorr-{roi}_run-{run}_events.tsv')
                    niiPa = newTSV.replace('events.tsv', f'run-{run}_desc-preproc_bold.nii.gz')
                   
                    stimNameET = path.join(outP, 'stimuli', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_apertures.nii.gz')
                            
                    if not path.exists(newTSV) or force:
                        with open(newTSV, 'w') as file:
                            
                            nii = nib.load(stimNameET)
                            TR = nii.header['pixdim'][4]
                            nT = nii.shape[3]
                            
                            # fill the events file
                            file.writelines('onset\tduration\tstim_file\tstim_file_index\n')
                            
                            for i in range(nT):
                                file.write(f'{i*TR:.3f}\t{TR:.3f}\t{stimNameET.split("/")[-1]}\t{i+1}\n')
                
    



