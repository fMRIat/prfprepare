#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:28:36 2021

@author: dlinhardt
"""

import copy
import json
import sys
from glob import glob
from os import makedirs, path

import bids
import neuropythy as ny
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.io import loadmat
from scipy.ndimage import binary_dilation


def die(*args):
    print(*args)
    sys.exit(1)


def note(*args):
    if verbose:
        print(*args)
    return None


def load_atlas(atlas, fsDir, sub, hemi, rois):
    if atlas == 'benson':
        areasP = path.join(fsDir, f'sub-{sub}', 'surf', f'{hemi}h.benson14_varea.mgz')
        if not path.exists(areasP):
            die(f'We could not find the benson atlas fiel: {areasP}')

        # load the label files
        areas = nib.load(areasP).get_fdata()[0, 0, :]

        # load the label area dependency
        mdl = ny.vision.retinotopy_model('benson17', f'{hemi}h')
        areaLabels = dict(mdl.area_id_to_name)
        areaLabels = {areaLabels[k]: k for k in areaLabels}
        labelNames = list(areaLabels.keys())

        if rois[0] == 'all':
            rois = labelNames

        atlasName = atlas

    elif atlas == 'wang':
        areasP = path.join(fsDir, f'sub-{sub}', 'surf', f'{hemi}h.wang15_mplbl.mgz')
        if not path.exists(areasP):
            die(f'We could not find the wang atlas file: {areasP}')

        # load the label files
        areas = nib.load(areasP).get_fdata()[0, 0, :]

        labelNames = ["Unknown",
                      "V1v", "V1d", "V2v", "V2d", "V3v", "V3d",
                      "hV4", "VO1", "VO2", "PHC1", "PHC2",
                      "V3a", "V3b", "LO1", "LO2", "TO1", "TO2",
                      "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5",
                      "SPL1", "hFEF"]
        areaLabels = {labelNames[k]: k for k in range(len(labelNames))}

        if rois[0] == 'all':
            rois = labelNames[1:]

        atlasName = atlas

    elif 'annot' in atlas:
        if hemi+'h.' in atlas:
            annotP = path.join(fsDir, f'sub-{sub}', 'customLabel', atlas)

            a,c,l = nib.freesurfer.io.read_annot(annotP)
            areas = a + 1
            areaLabels = {'Unknown':0} | {l[k].decode('utf-8'): k+1 for k in range(len(l))}

            if rois[0] == 'all':
                rois = list(areaLabels.keys())[1:]

            atlasName = atlas.split('.')[1]
        else:
            return [], [], [], []

    elif atlas in ['none', 'fullBrain']:
        labelNames = 'fullBrain'
        areaLabels = {'fullBrain': -1}
        rois = ['fullBrain']

    else:
        die('You specified a wrong atlas, please choose from [benson, wang, fs_custom]!')

    return areas, areaLabels, rois, atlasName


###############################################################################
def nii_to_surfNii(sub, sess, layout, bidsDir, subInDir, outP, fsDir, forceParams,
                   fmriprepLegacyLayout, average, output_only_average,
                   atlases, roisIn, analysisSpace, force, verbose):
    '''
    This function converts the surface _bold.func.gz files to 2D nifti2 files
    where every pixel contains one vertex timecourse. Different ROIs specified
    are merged into one mask and we output one nifti2 file containing all
    voxel data within any of the defined ROIs. This allows for minimising
    computation time since vertices contained in e.g. wang-V1 and benson-V1
    only have to be analysed once! Further, for ever atals and ROI we output
    one sidecar .json giving all information about which indices wihin the
    nifti file belongs to the ROI and which indicices in fs space correspond.

    Additionally the first timepoints are removed as defined in PrescanDuration
    as well as startScan in the _params.mat file.
    '''

    def note(*args):
        if verbose:
            print(*args)
        return None

    for hemi in ['l', 'r']:
        # first get the total number of vertices

        if analysisSpace == 'fsnative':

            nVertices = len(nib.freesurfer.io.read_geometry(path.join(fsDir,
                                                        f'sub-{sub}', 'surf',
                                                        f'{hemi.lower()}h.pial'))[0])

            # define the empty mask
            allROImask = np.zeros(nVertices)

            # find the merged mask
            # loop over all defined atlases
            for atlas in atlases:
                # load in the atlas
                areas, areaLabels, rois, atlasName = load_atlas(atlas, fsDir, sub, hemi, roisIn)

                # go for all given ROIs
                for roi in rois:
                    # if we want fullBrain change the mask to all ones
                    if roi == 'fullBrain':
                        allROImask = np.ones(allROImask.shape)

                    # else we adapt the mask for the roi
                    else:
                        # get labels associated with ROI
                        roiLabels = [value for key, value in areaLabels.items() if roi in key]

                        if not roiLabels:
                            note(f'We could not find {roi} in atlas {atlas}, continue...')
                            continue

                        # define the json name for

                        thisROImask = np.array([ar in roiLabels for ar in areas])
                        allROImask = np.any((allROImask, thisROImask), 0)

            # define the json files for the found mask and apply it to bold data
            # loop over all defined atlases
            for atlas in atlases:
                # load in the atlas
                areas, areaLabels, rois, atlasName = load_atlas(atlas, fsDir, sub, hemi, roisIn)

                # go for all given ROIs
                for roi in rois:
                    # if we want fullBrain change the mask to all ones
                    if roi == 'fullBrain':
                        allROImask = np.ones(allROImask.shape)

                    # else we adapt the mask for the roi
                    else:
                        # get labels associated with ROI
                        roiLabels = [value for key, value in areaLabels.items() if roi in key]

                        if not roiLabels:
                            continue

                        thisROImask = np.array([ar in roiLabels for ar in areas])

                        # define a list of all appliccable boldFiles
                        for sesI, ses in enumerate(sess):
                            boldFiles = []
                            funcOutP = path.join(outP, f'ses-{ses}', 'func')
                            makedirs(funcOutP, exist_ok=True)

                            tasks = layout.get_tasks(subject=sub, session=ses)

                            for task in tasks:
                                runs = layout.get_runs(subject=sub, session=ses, task=task)

                                # adapt for averaged runs
                                if average and len(runs) > 1:
                                    if output_only_average:
                                        runs = [''.join(map(str, runs)) + 'avg']
                                    else:
                                        runs.append(''.join(map(str, runs)) + 'avg')
                                for run in runs:
                                    boldFiles.append(f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi.upper()}_bold.nii.gz')

                            # define the json  for this specific atlas-roi combi for one subject and session
                            jsonP = path.join(funcOutP, 
                                        f'sub-{sub}_ses-{ses}_hemi-{hemi.upper()}_desc-{roi}-{atlasName}_maskinfo.json')
                            jsonI = {'atlas': atlasName,
                                     'roi': roi,
                                     'hemisphere': hemi,
                                     'thisHemiSize': int(allROImask.sum()),
                                     'boldFiles': boldFiles,
                                     'roiIndFsnative': np.where(thisROImask)[0].tolist(),
                                     'roiIndBold': np.where(thisROImask[allROImask])[0].tolist()
                                     }
                            if len(jsonI['roiIndFsnative']) != len(jsonI['roiIndBold']):
                                die('Something wrong with the Indices!!')
                            with open(jsonP, 'w') as fl:
                                json.dump(jsonI, fl, indent=4)

        elif  analysisSpace == 'volume':
            # load the GM mask from freesurfer for the receptive hemi
            ribbon = nib.load(path.join(fsDir, f'sub-{sub}', 'mri', f'{hemi}h.ribbon.mgz'))

            # dilate the mask and save it
            dilRibbonData = binary_dilation(ribbon.get_fdata())
            dilRibbon = nib.Nifti1Image(dilRibbonData, header=ribbon.header, affine=ribbon.affine)
            nib.save(dilRibbon, path.join(fsDir, f'sub-{sub}', 'mri', f'{hemi}h.dil_ribbon.mgz'))

            # load an example bold image
            boldref = nib.load(glob(path.join(subInDir, 'ses-*', 'func', 
                            f'sub-{sub}_ses-*_task-*_run-*_space-T1w_boldref.nii.gz'))[0])
            exampleBold = nib.load(glob(path.join(subInDir, 'ses-*', 'func', 
                            f'sub-{sub}_ses-*_task-*_run-*_space-T1w_desc-preproc_bold.nii.gz'))[0]).get_fdata()
            
            # resample the mask to bold space
            resDilRibbon = resample_from_to(dilRibbon, boldref, order=0)
            nib.save(resDilRibbon, path.join(fsDir, f'sub-{sub}', 'mri', f'{hemi}h.res_dil_ribbon.mgz'))
        
            allROImask = np.all((resDilRibbon.get_fdata().astype(bool), exampleBold.sum(-1)>0), 0)
            for sesI, ses in enumerate(sess):
                funcOutP = path.join(outP, f'ses-{ses}', 'func')
                makedirs(funcOutP, exist_ok=True)

                # define the json for this specific atlas-roi combi for one subject and session
                jsonP = path.join(funcOutP, 
                            f'sub-{sub}_ses-{ses}_hemi-{hemi.upper()}_desc-volume_maskinfo.json')
                jsonI = {'atlas': 'volume',
                        'roi': 'volume',
                        'hemisphere': hemi,
                        'thisHemiSize': int(allROImask.sum()),
                        'posBold': np.array(np.where(allROImask)).T.tolist()
                        }
                with open(jsonP, 'w') as fl:
                    json.dump(jsonI, fl, indent=4)
        
        else:
            die(f'Your analysisSpace {analysisSpace} is not supported! '
                'Please choose from [fsaverage, volume]')
        # now lets apply the merged mask to all bold files
        for sesI, ses in enumerate(sess):
            note(f'[nii_to_sufNii.py] Working on sub-{sub} ses-{ses} hemi-{hemi.upper()}')
            funcInP = path.join(subInDir, f'ses-{ses}', 'func')
            funcOutP = path.join(outP, f'ses-{ses}', 'func')

            tasks = layout.get_tasks(subject=sub, session=ses)
            for task in tasks:
                runs = layout.get_runs(subject=sub, session=ses, task=task)
                # adapt for averaged runs
                if average and len(runs) > 1:
                    runsOrig = copy.copy(runs)
                    if output_only_average:
                        runs = [''.join(map(str, runs)) + 'avg']
                    else:
                        runs.append(''.join(map(str, runs)) + 'avg')
                for run in runs:
                    # check if already exists, if not force skip
                    # if not path.exists(newNiiP) or force:

                    if 'av' not in str(run):
                        # name the output files
                        newNiiP = path.join(funcOutP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi.upper()}_bold.nii.gz')

                        # load the .gii in fsnative
                        if analysisSpace == 'fsnative':
                            if fmriprepLegacyLayout:
                                giiP = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsnative_hemi-{hemi.upper()}_bold.func.gii')
                            else:
                                giiP = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi.upper()}_space-fsnative_bold.func.gii')

                            # get the data data
                            data = nib.load(giiP).agg_data()

                        # or volume file in T1 space
                        elif analysisSpace == 'volume':
                            niiP = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii.gz')#'hemi-{hemi.upper()}_bold.func.gii')

                            # get the data data
                            data = nib.load(niiP).get_fdata()


                    else:
                        # name the output files
                        newNiiP = path.join(funcOutP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi.upper()}_bold.nii.gz')

                        datas = []
                        for r in runsOrig:
                            if analysisSpace == 'fsnative':
                                if fmriprepLegacyLayout:
                                    giiP = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{r}_space-fsnative_hemi-{hemi.upper()}_bold.func.gii')
                                else:
                                    giiP = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{r}_hemi-{hemi.upper()}_space-fsnative_bold.func.gii')

                                datas.append(nib.load(giiP).agg_data())

                            # or volume file in T1 space
                            elif analysisSpace == 'volume':
                                niiP = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii.gz')#'hemi-{hemi.upper()}_bold.func.gii')

                                # get the data data
                                datas.append(nib.load(niiP).get_fdata())

                        if len(datas) > 1:
                            # crop them to the same length for averaging
                            giiMinLength = min([g.shape[-1] for g in datas])
                            gii = [g[..., :giiMinLength] for g in datas]
                            # average the runs
                            data = np.mean(gii, 0)
                        else:
                            data = gii[0]

                    # apply the combined ROI mask
                    data = data[allROImask, :]

                    # get rid of volumes where the stimulus showed only blank (prescanDuration)
                    if forceParams:
                        paramsFile, task = forceParams
                        params = loadmat(path.join(bidsDir, 'sourcedata', 'vistadisplog', paramsFile),
                                         simplify_cells=True)
                    else:
                        if 'av' not in str(run):
                            params = loadmat(path.join(bidsDir, 'sourcedata', 'vistadisplog', f'sub-{sub}',
                                                       f'ses-{ses}', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_params.mat'),
                                             simplify_cells=True)
                        else:
                            params = loadmat(path.join(bidsDir, 'sourcedata', 'vistadisplog', f'sub-{sub}',
                                                       f'ses-{ses}', f'sub-{sub}_ses-{ses}_task-{task}_run-01_params.mat'),
                                             simplify_cells=True)

                    tr = params['params']['tr']

                    if 'prescanDuration' in params['params'].keys():
                        prescan = params['params']['prescanDuration']

                        if prescan > 0:
                            note(f'Removing {int(prescan/tr)} volumes from the beginning due to prescan')
                            data = data[:, int(prescan / tr):]

                    else:
                        prescan = 0

                    # remove volumes the stimulus was wating to start (startScan)
                    if 'startScan' in params['params'].keys():
                        startScan = params['params']['startScan']

                        if startScan  > 0:
                            note(f'Removing {int(startScan/tr)} volumes from the beginning due to startScan')
                            data = data[:, int(startScan / tr):]


                    # create and save new nii img
                    try:
                        apertures = np.array(glob(path.join(outP, 'stimuli', 'task-*_apertures.nii.gz')))
                        stimNii = nib.load(apertures[[f'task-{task}_' in ap for ap in apertures]].item())
                    except:
                        print(f'could not find task-{task} in {path.join(outP, "stimuli")}!')
                        continue

                    # trim data to stimulus length, gets rid of volumes when the
                    # scanner was running for longer than the task and is topped manually
                    stimLength = stimNii.shape[-1]
                    if data.shape[1] < stimLength:
                        die(f'For {path.basename(newNiiP)} the data is shorter than '
                            F'the simulus file ({data.shape[1]}<{stimLength})')
                    elif data.shape[1] > stimLength:
                        data = data[:, :stimLength]
                    else:
                        pass

                    # save the new nifti
                    newNii = nib.Nifti2Image(data[:, None, None, :].astype('float32'), affine=np.eye(4))
                    newNii.header['pixdim'] = stimNii.header['pixdim']
                    newNii.header['qoffset_x'] = 1
                    newNii.header['qoffset_y'] = 1
                    newNii.header['qoffset_z'] = 1
                    newNii.header['cal_max'] = 1
                    newNii.header['xyzt_units'] = 10
                    nib.save(newNii, newNiiP)
