#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:28:36 2021

@author: dlinhardt
"""

import nibabel as nib
from os import path, makedirs
import bids
import numpy as np
import neuropythy as ny
import argparse
from glob import glob
from scipy.io import loadmat
import sys

def nii_to_surfNii(sub, sess, layout, bidsDir, subInDir, outP, fsDir, forceParams,
                   fmriprepLegacyLayout, atlases, rois, etcorr, force, verbose):
    def die(*args):
        print(*args)
        sys.exit(1)
    def note(*args):
        if verbose: print(*args)
        return None

    if forceParams:
        paramsFile, task = forceParams
    
    # loop over all defined atlases
    for atlas in atlases:
        # load in the atlas
        if atlas == 'benson':
            lh_areasP = path.join(fsDir, f'sub-{sub}', 'surf', 'lh.benson14_varea.mgz')
            rh_areasP = path.join(fsDir, f'sub-{sub}', 'surf', 'rh.benson14_varea.mgz')
            if not path.exists(lh_areasP) or not path.exists(rh_areasP): exit(4)
            
            # load the label files
            lh_areas = nib.load(lh_areasP).get_fdata()[0,0,:]
            rh_areas = nib.load(rh_areasP).get_fdata()[0,0,:]
            
            # load the label area dependency
            mdl = ny.vision.retinotopy_model('benson17', 'lh')
            areaLabels = dict(mdl.area_id_to_name)
            areaLabels = { areaLabels[k]:k for k in areaLabels }
            labelNames = list(areaLabels.keys())
            
            if rois[0] == 'all':
                rois = labelNames
            
        elif atlas == 'wang':
            lh_areasP = path.join(fsDir, f'sub-{sub}', 'surf', 'lh.wang15_mplbl.mgz')
            rh_areasP = path.join(fsDir, f'sub-{sub}', 'surf', 'rh.wang15_mplbl.mgz')
            if not path.exists(lh_areasP) or not path.exists(rh_areasP): exit(4)
            
            # load the label files
            lh_areas = nib.load(lh_areasP).get_fdata()[0,0,:]
            rh_areas = nib.load(rh_areasP).get_fdata()[0,0,:]
            
            labelNames = ["Unknown",
                 "V1v", "V1d", "V2v", "V2d", "V3v", "V3d",
                 "hV4", "VO1", "VO2", "PHC1", "PHC2",
                 "V3A", "V3B", "LO1", "LO2", "TO1", "TO2",
                 "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5",
                 "SPL1", "hFEF"]            
            areaLabels = {labelNames[k]:k for k in range(len(labelNames))}
            
            if rois[0] == 'all':
                rois = labelNames[1:]
            
        else:
            die('You specified a wrong atlas, please choose from [benson, wang]!')
        
        # go for all given ROIs
        for roi in rois:
            if roi not in labelNames:
                continue
            # get labels associated with ROI
            roiLabels = [ value for key, value in areaLabels.items() if roi in key ]
                                                
            for sesI,ses in enumerate(sess):
                funcInP  = path.join(subInDir, f'ses-{ses}', 'func')
                funcOutP = path.join(outP, f'ses-{ses}', 'func')
                makedirs(funcOutP, exist_ok=True)
                       
                lh_mask = np.array([ l in roiLabels for l in lh_areas ])
                rh_mask = np.array([ l in roiLabels for l in rh_areas ])
                
                tasks = layout.get(subject=sub, session=ses, return_type='id', target='task')
                
                for task in tasks:
                    
                    runs = layout.get(subject=sub, session=ses, task=task, return_type='id', target='run')
                    
                    for run in runs:
                        
                        newNiiPL = path.join(funcOutP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-L_desc-{roi}-{atlas}_bold.nii.gz')
                        newNiiPR = path.join(funcOutP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-R_desc-{roi}-{atlas}_bold.nii.gz')
        
                        if not path.exists(newNiiPL) or not path.isfile(newNiiPR) or force:
                            
                            # load the .gii in fsnative
                            if fmriprepLegacyLayout:
                                # print('Reading fmriprep legacy files, space then hemi')
                                giiPL = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsnative_hemi-L_bold.func.gii')
                                giiPR = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsnative_hemi-R_bold.func.gii')
                            else:
                                # print('Reading fmriprep NON-legacy files, hemi then space')
                                giiPL = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-L_space-fsnative_bold.func.gii')
                                giiPR = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-R_space-fsnative_bold.func.gii')
                            giiImgL = nib.load(giiPL).agg_data()
                            giiImgR = nib.load(giiPR).agg_data()
                            
                            # apply V1 mask
                            verticesL = giiImgL[lh_mask,:]
                            verticesR = giiImgR[rh_mask,:]
                            
                            # get rid of volumes before stimulus actually started
                            if forceParams:
                                paramsFile, task = forceParams
                                params = loadmat(path.join(bidsDir, 'sourcedata', 'vistadisplog', paramsFile), 
                                                 simplify_cells=True)
                            else:  
                                params = loadmat(path.join(bidsDir, 'sourcedata', 'vistadisplog', f'sub-{sub}', 
                                                           f'ses-{ses}', f'sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_params.mat'), 
                                                 simplify_cells=True)
                                
                            if params['params']['prescanDuration']>0:
                                verticesL = verticesL[:,int(params['params']['prescanDuration']+1):]
                                verticesR = verticesR[:,int(params['params']['prescanDuration']+1):]
                            
                            # create and save new nii img
                            try:
                                apertures = np.array(glob(path.join(outP, 'stimuli', 'task-*.nii.gz')))
                                stimNii = nib.load(apertures[[f'task-{task}' in ap for ap in apertures]].item())
                            except:
                                print(f'could not find task-{task} in {path.join(outP, "stimuli")}!')
                                continue
                            
                            # save the left hemi
                            newNiiL = nib.Nifti2Image(verticesL[:,None,None,:].astype('float64'), affine=np.eye(4))
                            newNiiL.header['pixdim']     = stimNii.header['pixdim']
                            newNiiL.header['qoffset_x']  = 1
                            newNiiL.header['qoffset_y']  = 1
                            newNiiL.header['qoffset_z']  = 1
                            newNiiL.header['cal_max']    = 1
                            newNiiL.header['xyzt_units'] = 10
                            nib.save(newNiiL, newNiiPL)
                            
                            # save the right hemi
                            newNiiR = nib.Nifti2Image(verticesR[:,None,None,:].astype('float64'), affine=np.eye(4))
                            newNiiR.header['pixdim']     = stimNii.header['pixdim']
                            newNiiR.header['qoffset_x']  = 1
                            newNiiR.header['qoffset_y']  = 1
                            newNiiR.header['qoffset_z']  = 1
                            newNiiR.header['cal_max']    = 1
                            newNiiR.header['xyzt_units'] = 10
                            nib.save(newNiiR, newNiiPR)
                            
                            if etcorr:
                                outPET = outP.replace('/sub-', '_ET/sub-')
                                if path.isdir(outP):
                                    funcOutPET = path.join(outPET, f'ses-{ses}', 'func')
                                    makedirs(funcOutPET, exist_ok=True)
                                    newNiiPETL = path.join(funcOutPET, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-L_desc-{roi}_bold.nii.gz')
                                    newNiiPETR = path.join(funcOutPET, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-R_desc-{roi}_bold.nii.gz')
                                    nib.save(newNiiL, newNiiPETL)
                                    nib.save(newNiiR, newNiiPETR)
                                else:
                                    print('No eyetracker analysis-XX folder found!')
                    

