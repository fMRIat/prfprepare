#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:28:36 2021

@author: dlinhardt
"""

import nibabel as nib
from os import path
import bids
import numpy as np
import argparse
from glob import glob
import neuropythy as ny
import sys

def link_stimuli(sub, sess, layout, bidsDir, outP, atlases, rois, etcorr, force, verbose):
    def die(*args):
        print(*args)
        sys.exit(1)
    def note(*args):
        if verbose: print(*args)
        return None

    for atlas in atlases:
        if rois[0] == 'all':
            if atlas == 'benson':
                mdl = ny.vision.retinotopy_model('benson17', 'lh')
                areaLabels = dict(mdl.area_id_to_name)
                areaLabels = { areaLabels[k]:k for k in areaLabels }
                rois = list(areaLabels.keys())
            elif atlas == 'wang':
                rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d",
                        "hV4", "VO1", "VO2", "PHC1", "PHC2",
                        "V3A", "V3B", "LO1", "LO2", "TO1", "TO2",
                        "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5",
                        "SPL1", "hFEF"]
                
        # go for all given ROIs
        for roi in rois:
            
            for sesI,ses in enumerate(sess):
            
                tasks = layout.get(subject=sub, session=ses, return_type='id', target='task')  
            
                for task in tasks:
                    
                    apertures = np.array(glob(path.join(outP, 'stimuli', 'task-*.nii.gz')))
                    try:
                        stimName  = apertures[[f'task-{task}' in ap for ap in apertures]].item()
                    except:
                        continue
                    
                    
                    runs = layout.get(subject=sub, session=ses, task=task, return_type='id', target='run')
                    
                    for run in runs:
                        
                        # create events.tsv without ET
                        newTSV = path.join(outP, f'ses-{ses}', 'func', 
                                           f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv')
                        niiPa = newTSV.replace('_events.tsv', f'_desc-{roi}-{atlas}_bold.nii.gz')
                        
                        if not path.isfile(newTSV) or force:
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
                            outPET = outP.replace('/sub-', '_ET/sub-')
                            if path.isdir(outPET):
                                
                                newTSV = path.join(outPET, f'ses-{ses}', 'func', 
                                                   f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv')
                                niiPa = newTSV.replace('_events.tsv', f'_desc-{roi}-{atlas}_bold.nii.gz')
                               
                                stimNameET = path.join(outPET, 'stimuli', f'sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_apertures.nii.gz')
                                        
                                if not path.isfile(newTSV) or force:
                                    if not path.isfile(stimNameET):
                                        print(f'did not find {stimNameET}!')
                                        continue
                                        
                                    nii = nib.load(stimNameET)
                                    TR = nii.header['pixdim'][4]
                                    nT = nii.shape[3]
                                    
                                    # fill the events file
                                    with open(newTSV, 'w') as file:
                                        file.writelines('onset\tduration\tstim_file\tstim_file_index\n')
                                        
                                        for i in range(nT):
                                            file.write(f'{i*TR:.3f}\t{TR:.3f}\t{stimNameET.split("/")[-1]}\t{i+1}\n')
                                            
                            else:
                                print('No eyetracker analysis-XX folder found!')
                        
    



