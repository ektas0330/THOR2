#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:35:17 2023

@author: smartslab
"""

import fnmatch,os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
import copy 
import pickle
from scipy.spatial.transform import Rotation
import time
from persim import PersistenceImager
from scipy.stats import iqr
import yaml
import shutil
import json,h5py

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
import sys,argparse
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

def classifier_mlp_softmax(n_classes=17,objlayers= 1):
    classifier = Sequential()
    classifier.add(Dense(512, input_shape = (1024*objlayers,)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(256))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(128))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(64))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))
    
    classifier.add(Dense(n_classes))
    classifier.add(BatchNormalization())
    classifier.add(Activation('softmax'))
    
    return classifier


def scaleObjectPCD(pcd,scaleFactor):
    scaled = copy.deepcopy(pcd)
    scaled.scale(scaleFactor,center=[0,0,0])
    return scaled


def getObjectPCD(pcd,objidx,label,color):
    label1d = np.reshape(label,(np.shape(label)[0]*np.shape(label)[1],))
    color1d = np.reshape(color,(np.shape(label)[0]*np.shape(label)[1],))/255
    idxes = np.where(label1d==objidx)
    allpts = np.asarray(pcd.points)
    objpts = allpts[idxes]
    colorpts = np.expand_dims(color1d[idxes],axis=1)
    objptscolor = np.concatenate((colorpts,np.zeros_like(colorpts),np.zeros_like(colorpts)),axis=1)
    objpcd = o3d.geometry.PointCloud()
    objpcd.points = o3d.utility.Vector3dVector(objpts)
    objpcd.colors = o3d.utility.Vector3dVector(objptscolor)
    return objpcd

def getObjectOcclusionColors(badidxes, objidx,label,color):
    editedlabel=copy.deepcopy(label)
    editedlabel[badidxes] = -1 
    label1d = np.reshape(editedlabel,(np.shape(label)[0]*np.shape(label)[1],))
    color1d = np.reshape(color,(np.shape(label)[0]*np.shape(label)[1],))/255
    idxes = np.where(label1d==objidx)
    colorpts = np.expand_dims(color1d[idxes],axis=1)
    objptscolor = np.concatenate((colorpts,np.zeros_like(colorpts),np.zeros_like(colorpts)),axis=1)
    return objptscolor
    
def findContour(objidx,label):
    binimg = np.expand_dims(255*np.where(label==objidx,1,0),axis=2).astype('uint8')
    img = np.concatenate((binimg,binimg,binimg),axis=2)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(binimg,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea)
    return np.squeeze(cnt[-1])[:,::-1] #because opencv flips rows and columns



def checkOccludeeContour(contour,objidx,label,depth):
    boundary = np.zeros_like(label)
    for i in range(np.shape(contour)[0]):
        x,y = contour[i,:]
        if x == 719:
            boundary[x,y] = 255 ## if object is cut
        elif y == 1279:
            boundary[x,y] = 255
        else:
            if label[x-1,y] > 0 and label[x-1,y] != objidx:
                if depth[x-1,y] < depth[x,y]:
                    boundary[x,y] = 255
            if label[x+1,y] > 0 and label[x+1,y] != objidx:
                if depth[x+1,y] < depth[x,y]:
                    boundary[x,y] = 255
            if label[x,y-1] > 0 and label[x,y-1] != objidx:
                if depth[x,y-1] < depth[x,y]:
                    boundary[x,y] = 255
            if label[x,y+1] > 0 and label[x,y+1] != objidx:
                if depth[x,y+1] < depth[x,y]:
                    boundary[x,y] = 255
    return boundary

def rotateToFlatForLayering(pcd):
    pcdpts = np.asarray(pcd.points)[:-2,:]
    bbox = o3d.geometry.OrientedBoundingBox()
    bboxresult = bbox.create_from_points(o3d.utility.Vector3dVector(pcdpts))#,robust=True)
    #o3d.visualization.draw([pcd,bboxresult])
    Rnew = np.transpose(bboxresult.R)
    pcd.rotate(Rnew)
    #now 2d using opencv


    #the angle is wrt to point with highest y. cw rotation of x axis until it meets an edge of bb
    w2,h2,angle2 = get2dboundingboxXYEfficient(np.asarray(pcd.points)[:-2,:])
    #w2,h2,angle2 = get2dboundingboxXY(np.asarray(pcd.points)[:-2,:])
    
    if h2 < w2:
        angles = [0,0, (angle2*np.pi)/180]
        R2dxy = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dxy)
    else:
        angle2 = 90-angle2
        angles = [0,0,-(angle2*np.pi)/180]
        R2dxy = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dxy)    


    w1,h1,angle1 = get2dboundingboxYZEfficient(np.asarray(pcd.points)[:-2,:])
    #w1,h1,angle1 = get2dboundingboxYZ(np.asarray(pcd.points)[:-2,:])

    if h1 < w1:
        angles = [(angle1*np.pi)/180,0,0]
        R2dyz = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dyz)
    else:
        angle1 = 90-angle1
        angles = [-(angle1*np.pi)/180,0,0]
        R2dyz = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dyz)
        
    campos = np.asarray(pcd.points)[-1,:]
    
    if campos[2] > 0:
        Rtemp = o3d.geometry.get_rotation_matrix_from_xyz([np.pi,0,0])
        pcd.rotate(Rtemp)
    
    pts = np.asarray(pcd.points)[:-2,:]
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bboxresult = bbox.create_from_points(o3d.utility.Vector3dVector(pts))#,robust=True)
    extent = bboxresult.get_extent()  
    
    return pcd,extent

def get2dboundingboxXYEfficient(points):
    final = my_scatter_plot_xy(points)
    final = final.astype(np.uint8) 
    kernel = np.ones((11, 11), np.uint8)
    #cv2.imwrite('myplot.png',final)
    # Using cv2.erode() method 
    final = cv2.erode(final, kernel, cv2.BORDER_REFLECT) 

    
    _,thresh = cv2.threshold(final,127,255,cv2.THRESH_BINARY_INV)

    # contours,_ = cv2.findContours(thresh.copy(), 1, 1) # not copying here will throw an error
    # rect = cv2.minAreaRect(contours[-1]) # basically you can feed this rect into your classifier
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt[-1]) # basically you can feed this rect into your classifier
    (x,y),(w,h), a = rect # a - angle
    #print(a)
    #print(w,h)

    return w,h,a

def get2dboundingboxYZEfficient(points):
    final = my_scatter_plot_yz(points)
    final = final.astype(np.uint8) 
    #cv2.imwrite('myplot.png',final)
    kernel = np.ones((11, 11), np.uint8)
      
    # Using cv2.erode() method 
    final = cv2.erode(final, kernel, cv2.BORDER_REFLECT) 
    
    
    _,thresh = cv2.threshold(final,127,255,cv2.THRESH_BINARY_INV)
    
    # contours,_ = cv2.findContours(thresh.copy(), 1, 1) # not copying here will throw an error
    # rect = cv2.minAreaRect(contours[-1]) # basically you can feed this rect into your classifier
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt[-1]) # basically you can feed this rect into your classifier

    (x,y),(w,h), a = rect # a - angle
    #print(a)
    return w,h,a

def my_scatter_plot_xy(points):
    nx,ny = (224,224)
    xmin = np.min(points[:,0])
    ymin = np.min(points[:,1])
    img = 255*np.ones((nx,ny))
    
    x = np.linspace(xmin - 0.1,xmin+0.9,nx)
    y = np.linspace(ymin-0.1,ymin+0.9,ny)
    xbins = np.digitize(points[:,0],x)
    ybins = len(y) - np.digitize(points[:,1],y)
    
    for i in range(len(points)):
        img[ybins[i],xbins[i]] = 0
    
    return img

def my_scatter_plot_yz(points):
    ny,nz = (224,224)
    ymin = np.min(points[:,1])
    zmin = np.min(points[:,2])
    img = 255*np.ones((ny,nz))
    
    y = np.linspace(ymin - 0.1,ymin+0.9,ny)
    z = np.linspace(zmin-0.1,zmin+0.9,nz)
    ybins = np.digitize(points[:,1],y)
    zbins = len(z) - np.digitize(points[:,2],z)
    
    for i in range(len(points)):
        img[zbins[i],ybins[i]] = 0
    
    return img   


def roundinXYZ(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,0] = np.round(pcdpts[:,0],2)
    pcdpts[:,1] = np.round(pcdpts[:,1],2)
    pcdpts[:,2] = np.round(pcdpts[:,2],1)
    return pcdpts

def trZMinusCam(pcd):
    pts = np.asarray(pcd.points)[:-2,:]
    pcd.translate([0,0,-np.min(pts[:,2])])
    return pcd

def trYMinusCam(pcd):
    pts = np.asarray(pcd.points)[:-2,:]
    pcd.translate([0,-np.min(pts[:,1]),0])
    return pcd

def trXMinusCam(pcd):
    pts = np.asarray(pcd.points)[:-2,:]
    pcd.translate([-np.min(pts[:,0]),0,0])
    return pcd

def checkNeedToFlipMinusCam(flatpcd):
    flatpts = np.asarray(flatpcd.points)[:-2,:]
    xmin = np.min(flatpts[:,0])
    ymin = np.min(flatpts[:,1])
    zmin = np.min(flatpts[:,2])
    points = flatpts - [xmin,ymin,zmin]
    
    xmax = np.max(points[:,0])                
    threecolors = np.asarray(flatpcd.colors)[:-2,:]
    colors = threecolors[:,0]
    bins = np.arange(0,xmax+0.015,0.01)
    #print(bins)
    xes = copy.deepcopy(points[:,0])
    #print(np.max(xes))
    points[:,0] = bins[np.digitize(xes,bins,right=True)]

    redptsinlastbins = np.count_nonzero(colors[np.where(points[:,0]== bins[-1])])+np.count_nonzero(colors[np.where(points[:,0]== bins[-2])])
    redptsinfirstbins = np.count_nonzero(colors[np.where(points[:,0]== bins[0])])+np.count_nonzero(colors[np.where(points[:,0]== bins[1])])
    
    if redptsinfirstbins > redptsinlastbins:
        return True
    else:
        return False

def getZs(pcdpts):
    zlist = sorted(np.unique(pcdpts[:,2]))
    zs = {}
    for idx,num in enumerate(zlist):
        zs[idx] = num
    return zs

def getLayer(pcdpts,zs,layeridx):
    return pcdpts[np.where(pcdpts[:,2] == zs[layeridx])]

def computePDBinningNo2DTranslation(pcd):
    #binning
    bins = np.arange(0,0.775,0.025)
    xes = copy.deepcopy(pcd[:,0])
    pcd[:,0] = bins[np.digitize(xes,bins,right=True)]
    xesnew = np.unique(pcd[:,0])
    newpts = []
    dgm = []
    for idx,x in enumerate(xesnew):
        ymax = np.max(pcd[np.where(pcd[:,0] == x)][:,1])
        ymin = np.min(pcd[np.where(pcd[:,0] == x)][:,1])
        #print(y)
        dgm.append((x,x+ymax-ymin,0))
    #noisepts = np.concatenate((np.expand_dims(bins,axis=1),np.expand_dims(bins,axis=1),np.zeros((len(bins),1))),axis=1)
    #return np.concatenate((np.asarray(dgm),noisepts),axis=0)
    return np.asarray(dgm)

def getFeatureNewPad(pis,maxLayers):
    features = np.reshape(pis[0],(1,682))
    for l in range(1,maxLayers):
        if l in pis:
            features = np.concatenate((features, np.reshape(pis[l],(1,682))),axis=1)
        else:
            features = np.concatenate((features, np.zeros((1,682))),axis=1)

    return features

def mygetCamPosViewingDirection(pcd):
    points = np.asarray(pcd.points)
    cam2tr = points[-1,:]
    cam1tr = points[-2,:]
    direction = cam2tr-cam1tr
    unitvec = direction/np.linalg.norm(direction)
    return cam2tr, unitvec       


def orientCamBottom(pcd):
    campos = np.asarray(pcd.points)[-1,:] 
    if campos[2] > 0:
        Rtemp = o3d.geometry.get_rotation_matrix_from_xyz([np.pi,0,0])
        pcd.rotate(Rtemp)
    return pcd
        
def filterDgm(dgm,thresh):
    newdgm = []
    for i in range(len(dgm)):
        if dgm[i,1] - dgm[i,0] <= thresh:
            newdgm.append((dgm[i,0],dgm[i,1],0))
    return np.asarray(newdgm)


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

def getRedColoredPoints(pcdorg,objptcolors):
    pts = np.asarray(pcdorg.points)
    return pts[np.where(objptcolors[:,0]== 1)], objptcolors[np.where(objptcolors[:,0]== 1)]

def separateRedBlackPoints(pcdorg,objptcolors):
    pts = np.asarray(pcdorg.points)
    return pts[np.where(objptcolors[:,0]== 1)], objptcolors[np.where(objptcolors[:,0]== 1)], pts[np.where(objptcolors[:,0]!= 1)], objptcolors[np.where(objptcolors[:,0]!= 1)] 

def separateRedBlackPointsWithColors(pcdorg,objptcolors,actualpcdcolors):
    pts = np.asarray(pcdorg.points)
    return pts[np.where(objptcolors[:,0]== 1)], objptcolors[np.where(objptcolors[:,0]== 1)], actualpcdcolors[np.where(objptcolors[:,0]== 1)], pts[np.where(objptcolors[:,0]!= 1)], objptcolors[np.where(objptcolors[:,0]!= 1)], actualpcdcolors[np.where(objptcolors[:,0]!= 1)]

    
def loadMLP(modeldir,layers,model_type):
    model = classifier_mlp_softmax(17,layers)

    model.load_weights(modeldir + '/mlp_'+model_type+'_' + str(l)+'layers.hdf5')
    return model

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


def maintest1(args.videodir,args.environment,args.category,args.separation,args.light, args.modeldir):
        
    if not os.path.exists('./testing/prediction_probabilities/'):
        os.makedirs('./testing/prediction_probabilities/')
        
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

        if not os.path.exists('./testing/prediction_probabilities/'+video):
            os.makedirs('./testing/prediction_probabilities/'+video)

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
                        depthnp = cv2.imread(videodir +video+'/images/'+fileid+'_depth.png',-1)
                        contour = findContour(idx,label)
                        boundary = checkOccludeeContour(contour,idx,label,depthnp) ### depth with all objects here
                        
                        ## now get the depth of just the object of interest and create its point cloud
                        depthnp[np.where(label!=idx)] = 0
                        depthnp[np.where(depthnp > 2500)] = 0
                        depthnp[np.where(depthnp < 260)] = 0
                        cv2.imwrite(videodir +video+'/temp/'+fileid+'_depth_idx_'+str(idx)+'.png',depthnp)
                        
                        ##finding bad pixels inside object i.e. where depth is 0
                        depthnp[np.where(label!=idx)] = -1 ###note depth is of type uint16 ....here it actually gets assigned to 65535 but i dont care
                        badidxes = np.where(depthnp == 0)
                        
                        objptcolors = getObjectOcclusionColors(badidxes,idx,label,boundary) #red color points in object pcd are occlusion boundary
                        depth = o3d.io.read_image(videodir +video+'/temp/'+fileid+'_depth_idx_'+str(idx)+'.png')
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,depth_trunc = 2.5,convert_rgb_to_intensity = False)
                        pcdorg = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam.intrinsic)#,cam.extrinsic)
                        pcdactualcolors = copy.deepcopy(np.asarray(pcdorg.colors))

                        pcdorg.colors = o3d.utility.Vector3dVector(objptcolors)
                        redpts, redcolors, redcolorsreal, blackpts,blackcolors, blackcolorsreal = separateRedBlackPointsWithColors(pcdorg, objptcolors, pcdactualcolors)
                        
                        pcdtodown = o3d.geometry.PointCloud()
                        pcdtodown.points = o3d.utility.Vector3dVector(blackpts)
                        pcdtodown.colors = o3d.utility.Vector3dVector(blackcolorsreal)

                        downpcd = pcdtodown.voxel_down_sample(voxel_size=0.003)
                        
                        ##add red points later, not here so that they don't go away in outlier removal
                        blackdownpcd,ind = downpcd.remove_radius_outlier(nb_points=220,radius=0.05)                        
                        downedblackcolorsreal = copy.deepcopy(np.asarray(blackdownpcd.colors))
                        
                        
                        downpcdpts = np.asarray(blackdownpcd.points)
                        downpcdcolors = np.zeros_like(downpcdpts)
                        downpcdptsplusred = np.concatenate((downpcdpts,redpts),axis=0)
                        downpcdcolorsplusred = np.concatenate((downpcdcolors,redcolors),axis=0)
                        downpcdcolorsplusredreal = np.concatenate((downedblackcolorsreal,redcolorsreal),axis=0)
                        
                        objpcd = o3d.geometry.PointCloud()
                        objpcd.points = o3d.utility.Vector3dVector(downpcdptsplusred)
                        objpcd.colors = o3d.utility.Vector3dVector(downpcdcolorsplusred)   
                            
                        pts = np.asarray(objpcd.points)
                        cam1 = np.expand_dims(np.asarray([0,0,-0.1]),axis=0)
                        cam2 = np.expand_dims(np.asarray([0,0,0]),axis=0) #i found that test time cam position is 0,0,0
                        pts = np.concatenate((pts,cam1,cam2),axis=0)
                        colors = np.asarray(objpcd.colors)
    
                        camcol = np.expand_dims(np.asarray([0,0,0]),axis=0)
                        colors = np.concatenate((colors,camcol,camcol),axis=0)
                        realcolors = np.concatenate((downpcdcolorsplusredreal,camcol,camcol),axis=0)                    
                        #need to do the above to maintain red color of occluded points 
                        downpcdCam = o3d.geometry.PointCloud()
                        downpcdCam.points = o3d.utility.Vector3dVector(pts)
                        downpcdCam.colors = o3d.utility.Vector3dVector(colors)
                        
                        downpcdCamRealColors = o3d.geometry.PointCloud()
                        downpcdCamRealColors.colors = o3d.utility.Vector3dVector(realcolors)
                        downpcdCamRealColors.points = o3d.utility.Vector3dVector(pts)
                        
    
                        rotatedpcd,extent = rotateToFlatForLayering(downpcdCam) 
                        rotatedpcdreal,_ = rotateToFlatForLayering(downpcdCamRealColors)
                        
                        scaledpcd = scaleObjectPCD(rotatedpcd,2.5) #scaling will scale cam positions appropriately   
                        scaledpcdreal = scaleObjectPCD(rotatedpcdreal,2.5) #scaling will scale cam positions appropriately    
                        
                        needtoflip = checkNeedToFlipMinusCam(scaledpcd)
                        if needtoflip:
                            Rflip = o3d.geometry.get_rotation_matrix_from_xyz([0,0,np.pi])
                            scaledpcd.rotate(Rflip)
                            scaledpcdreal.rotate(Rflip)
                        else:
                            donothing=1
                        
                        trpcd = trXMinusCam(trYMinusCam(trZMinusCam(scaledpcdreal))) ##i no longer need the pcd with occlusion red black colors
                        rotatedpcd = orientCamBottom(trpcd)
                        
                        flatpcd = copy.deepcopy(rotatedpcd)
                        
                        R45 = o3d.geometry.get_rotation_matrix_from_xyz([0,-np.pi/4,0])
                        rotatedpcd.rotate(R45)    
                        
                        finalpcd = trXMinusCam(trYMinusCam(trZMinusCam(rotatedpcd)))
            
                        pcdpts = np.asarray(finalpcd.points)[:-2,:]
                        ptscolors = np.asarray(finalpcd.colors)[:-2,:]
                        rounded = roundinXYZ(pcdpts)
                        zs = getZs(rounded)
                        
                        pis = {}
                        for key,value in zs.items():
                            layer = getLayer(rounded,zs,key)
                            dgm = computePDBinningNo2DTranslation(layer)
        
                            pers = dgm[:,1] - dgm[:,0]
                            if (pers > 0.75).any():
                                print('pers range issue')
                            img = pimgr.transform([dgm[:,0:2]])
                            pis[key] = img[0]
        
                        topsfeature= getFeatureNewPad(pis,maxLayers)
                        
                        l = maxLayers  

                        probs = model.predict(np.nan_to_num(topsfeature[:,:1024*(l+1)]))[0]
                    
                        np.save('./testing/prediction_probabilities/'+video+'/'+ fileid+'_idx'+str(idx)+'_tops_pred_probs.npy',probs)
                                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videodir')
    parser.add_argument('--environment')
    parser.add_argument('--category')
    parser.add_argument('--separation')
    parser.add_argument('--light')  
    parser.add_argument('--modeldir')   
    args = parser.parse_args()
    maintest1(args.videodir,args.environment,args.category,args.separation,args.light, args.modeldir)
        
        
        
