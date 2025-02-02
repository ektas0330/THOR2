#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""

import fnmatch,os
import cv2
import numpy as np
import open3d as o3d

import yaml
import shutil


def maxOfTop1overTOPSAndTOPS2(topscolorprob,topsonlyprob):

    topscolorpred = np.argmax(topscolorprob)
    topscolormax = topscolorprob[0,topscolorpred]
    topspred = np.argmax(topsonlyprob)
    topsmax = topsonlyprob[0,topspred]
    
    maxprob = max(topscolormax, topsmax)
    
    if maxprob == topscolormax:
        return topscolorpred
    elif maxprob == topsmax:
        return topspred
    else:
        return NotImplementedError 

def getGroundTruthFromYAML(file):
    with open(file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            labels = {}
            for key,value in data.items():
                labels[value['label']] = key
        except yaml.YAMLError as exc:
            print(exc)
    return labels            

def returnvideolist(environment,category,separation,light):

    o1kitchenl1 = [41,42,176,177]
    o1foodl1 = [32,33,167,168]
    o1toolsl1 =[50,51,185,186]
    o2kitchenl1 = [44,45,179,180]
    o2foodl1 = [35,36,170,171]
    o2toolsl1 = [53,54,188,189]
    o3kitchenl1 = [46,48,182,183]
    o3foodl1 = [38,39,173,174]
    o3toolsl1 = [56,57,191,192]

    o1kitchenl2 = [68,69,203,204]
    o1foodl2 = [77,78,212,213]
    o1toolsl2 =[59,60,194,195]
    o2kitchenl2 = [71,72,206,207]
    o2foodl2 = [80,81,215,216]
    o2toolsl2 = [62,63,197,198]
    o3kitchenl2 = [74,75,209,210]
    o3foodl2 = [83,84,218,219]
    o3toolsl2 = [65,66,200,201]


    if light == '1':
        if category == 'kitchen' and separation == 'level1':
            intvideolist = o1kitchenl1
        elif category == 'kitchen' and separation == 'level2':
            intvideolist = o2kitchenl1
        elif category == 'kitchen' and separation == 'level3':
            intvideolist = o3kitchenl1
        elif category == 'kitchen' and separation == 'alllevels':
            intvideolist = o1kitchenl1 + o2kitchenl1 + o3kitchenl1
        
        elif category == 'food' and separation == 'level1':
            intvideolist = o1foodl1
        elif category == 'food' and separation == 'level2':
            intvideolist = o2foodl1
        elif category == 'food' and separation == 'level3':
            intvideolist = o3foodl1
        elif category == 'food' and separation == 'alllevels':
            intvideolist = o1foodl1 + o2foodl1 + o3foodl1
            
        elif category == 'tools' and separation == 'level1':
            intvideolist = o1toolsl1
        elif category == 'tools' and separation == 'level2':
            intvideolist = o2toolsl1
        elif category == 'tools' and separation == 'level3':
            intvideolist = o3toolsl1
        elif category == 'tools' and separation == 'alllevels':
            intvideolist = o1toolsl1 + o2toolsl1 + o3toolsl1
            
        elif category == 'all' and separation == 'level1':
            intvideolist = o1toolsl1 + o1foodl1 + o1kitchenl1
        elif category == 'all' and separation == 'level2':
            intvideolist = o2toolsl1 + o2foodl1 + o2kitchenl1
        elif category == 'all' and separation == 'level3':
            intvideolist = o3toolsl1 + o3foodl1 + o3kitchenl1
        elif category == 'all' and separation == 'alllevels':
            intvideolist = o1kitchenl1 + o1foodl1+o1toolsl1 +o2kitchenl1 + o2foodl1+o2toolsl1+o3kitchenl1 + o3foodl1+o3toolsl1
        else:
            raise NotImplementedError
    elif light == '2':
        if category == 'kitchen' and separation == 'level1':
            intvideolist = o1kitchenl2
        elif category == 'kitchen' and separation == 'level2':
            intvideolist = o2kitchenl2
        elif category == 'kitchen' and separation == 'level3':
            intvideolist = o3kitchenl2
        elif category == 'kitchen' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o2kitchenl2 + o3kitchenl2
        
        elif category == 'food' and separation == 'level1':
            intvideolist = o1foodl2
        elif category == 'food' and separation == 'level2':
            intvideolist = o2foodl2
        elif category == 'food' and separation == 'level3':
            intvideolist = o3foodl2
        elif category == 'food' and separation == 'alllevels':
            intvideolist = o1foodl2 + o2foodl2 + o3foodl2
            
        elif category == 'tools' and separation == 'level1':
            intvideolist = o1toolsl2
        elif category == 'tools' and separation == 'level2':
            intvideolist = o2toolsl2
        elif category == 'tools' and separation == 'level3':
            intvideolist = o3toolsl2
        elif category == 'tools' and separation == 'alllevels':
            intvideolist = o1toolsl2 + o2toolsl2 + o3toolsl2
            
        elif category == 'all' and separation == 'level1':
            intvideolist = o1toolsl2 + o1foodl2 + o1kitchenl2
        elif category == 'all' and separation == 'level2':
            intvideolist = o2toolsl2 + o2foodl2 + o2kitchenl2
        elif category == 'all' and separation == 'level3':
            intvideolist = o3toolsl2 + o3foodl2 + o3kitchenl2
        elif category == 'all' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o1foodl2+o1toolsl2 +o2kitchenl2 + o2foodl2+o2toolsl2+o3kitchenl2 + o3foodl2+o3toolsl2
        else:
            raise NotImplementedError
    elif light == 'both':
        if category == 'kitchen' and separation == 'level1':
            intvideolist = o1kitchenl1+o1kitchenl2
        elif category == 'kitchen' and separation == 'level2':
            intvideolist = o2kitchenl1+o2kitchenl2
        elif category == 'kitchen' and separation == 'level3':
            intvideolist = o3kitchenl1+o3kitchenl2
        elif category == 'kitchen' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o2kitchenl2 + o3kitchenl2 + o1kitchenl1 + o2kitchenl1 + o3kitchenl1
        
        elif category == 'food' and separation == 'level1':
            intvideolist = o1foodl1+o1foodl2
        elif category == 'food' and separation == 'level2':
            intvideolist = o2foodl2+o2foodl1
        elif category == 'food' and separation == 'level3':
            intvideolist = o3foodl2+o3foodl1
        elif category == 'food' and separation == 'alllevels':
            intvideolist = o1foodl2 + o2foodl2 + o3foodl2+o1foodl1 + o2foodl1 + o3foodl1
            
        elif category == 'tools' and separation == 'level1':
            intvideolist = o1toolsl2+o1toolsl1
        elif category == 'tools' and separation == 'level2':
            intvideolist = o2toolsl2+o2toolsl1
        elif category == 'tools' and separation == 'level3':
            intvideolist = o3toolsl2+o3toolsl1
        elif category == 'tools' and separation == 'alllevels':
            intvideolist = o1toolsl2 + o2toolsl2 + o3toolsl2+o1toolsl1 + o2toolsl1 + o3toolsl1
            
        elif category == 'all' and separation == 'level1':
            intvideolist = o1toolsl2 + o1foodl2 + o1kitchenl2 + o1toolsl1 + o1foodl1 + o1kitchenl1
        elif category == 'all' and separation == 'level2':
            intvideolist = o2toolsl2 + o2foodl2 + o2kitchenl2 + o2toolsl1 + o2foodl1 + o2kitchenl1
        elif category == 'all' and separation == 'level3':
            intvideolist = o3toolsl2 + o3foodl2 + o3kitchenl2 + o3toolsl1 + o3foodl1 + o3kitchenl1
        elif category == 'all' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o1foodl2+o1toolsl2 +o2kitchenl2 + o2foodl2+o2toolsl2+o3kitchenl2 + o3foodl2+o3toolsl2 + o1kitchenl1 + o1foodl1+o1toolsl1 +o2kitchenl1 + o2foodl1+o2toolsl1+o3kitchenl1 + o3foodl1+o3toolsl1
            
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
        
    if environment == 'warehouse':    
        intvideolist = [x for x in intvideolist if x < 167]
    elif environment == 'lounge':
        intvideolist = [x for x in intvideolist if x > 166]
    elif environment == 'both':
        intvideolist = intvideolist
    else:
        raise NotImplementedError

    return intvideolist


def maintest1(args.videodir,args.environment,args.category,args.separation,args.light, args.model_dir):
        
    if not os.path.exists('./testing/predictions/'):
        os.makedirs('./testing/predictions/')
        
    w = 1280
    h = 720
    fx = 920.458618164062
    fy = 921.807800292969
    ppx = 626.486083984375
    ppy = 358.848205566406
    cx = ppx
    cy = ppy
    

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx,fy, cx, cy)
    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic

    
    intvideolist = returnvideolist(environment,category,separation,light)
    

    videolist = [str(x) for x in intvideolist]
    seq = 'tops_all_layers'
    model_type = 'all'
    maxLayers = 7
    object_list = ['potted_meat_can', 'screw_driver', 'padlock', 'mug', 'tomato_soup', 'mustard_bottle', 'bowl', 'foam_brick', 'scissors', 'bleach_cleanser', 'tennis_ball', 'spoon', 'pitcher_base', 'clamp', 'plate', 'hammer', 'gelatin_box']

    printnames = False
    refnormals = np.asarray([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])


    model = loadMLP(loadMLP('./mlp_models/'+seq,maxLayers,model_type)



    for video in videolist:

        true = []
        allpreds = []

        featuresdict = {}

        instances = sorted(fnmatch.filter(os.listdir(videodir +video+'/images/'),'*.yaml'))
        for i in instances:
            print(i, ' in ', video)

            fileid = i.split('_')[0]
            label = cv2.imread(videodir +video+'/images/'+fileid+'_labels.png',0)
            rgb = o3d.io.read_image(videodir +video+'/images/'+fileid+'_rgb.png')
            groundtruthdict = getGroundTruthFromYAML(videodir +video+'/images/'+fileid+'_poses.yaml')
            labelidxes = [i for i in np.unique(label) if i > 0]
            
            for idx in labelidxes:
                if groundtruthdict[idx] not in ['ice_cream','hot_sauce','chips_can']:
                    try: 
                        tops2probs = np.load('./testing/prediction_probabilities/'+video+'/'+ fileid+'_idx'+str(idx)+'_tops2_pred_probs.npy')
                        topsprobs = np.load('./testing/prediction_probabilities/'+video+'/'+ fileid+'_idx'+str(idx)+'_tops_pred_probs.npy')
                        pred = maxOfTop1overTOPSAndTOPS2(tops2probs,topsprobs)
                        
                        allpreds.append(pred)
                        
                        true.append(object_list.index(groundtruthdict[idx]))
                        
        with open('./testing/predictions/'+video+'.txt', "w") as output:
            output.write(str(allpreds)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videodir')
    parser.add_argument('--environment')
    parser.add_argument('--category')
    parser.add_argument('--separation')
    parser.add_argument('--light')  
    parser.add_argument('--probs_dir')   
    args = parser.parse_args()
    maintest2(args.videodir,args.environment,args.category,args.separation,args.light, args.probs_dir)
 
        
        
