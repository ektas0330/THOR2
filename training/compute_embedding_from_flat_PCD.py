#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""

import cv2
import numpy as np
import open3d as o3d
import fnmatch,os
import matplotlib.pyplot as plt
import copy 
from gtda.plotting import plot_point_cloud
import pickle
from scipy.spatial.transform import Rotation



def removeNANs(pcd):
    a = np.asarray(pcd.points)
    pts = a[~np.isnan(a).any(axis=1)]
    newpcd = o3d.PointCloud()
    newpcd.points = o3d.utility.Vector3dVector(pts)
    return newpcd

def roundinXYZ(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,0] = np.round(pcdpts[:,0],2)
    pcdpts[:,1] = np.round(pcdpts[:,1],2)
    pcdpts[:,2] = np.round(pcdpts[:,2],1)
    return pcdpts

def getZs(pcdpts):
    zlist = sorted(np.unique(pcdpts[:,2]))
    zs = {}
    for idx,num in enumerate(zlist):
        zs[idx] = num
    return zs

def getColorLayer(ptscolors, pcdpts,zs,layeridx):
    layer = pcdpts[np.where(pcdpts[:,2] == zs[layeridx])]
    layercolor = ptscolors[np.where(pcdpts[:,2] == zs[layeridx])]
    roundedlayercolor = roundpcdcolor(layercolor)
    
    return layer,roundedlayercolor


def computeColorEmbeddingNo2DTranslation(pcd, pcdcolor, membership,nnodes):
    bins = np.arange(0,0.775,0.025)
    xes = copy.deepcopy(pcd[:,0])
    pcd[:,0] = bins[np.digitize(xes,bins,right=True)]
    xesnew = np.unique(pcd[:,0])
    
    colormatrix = []
    for x in bins:
        if x not in xesnew:
            colorprop = np.zeros((nnodes,))
            colormatrix.append(colorprop)
        else:
            colorprop = np.zeros((nnodes,))
            colors, colorcounts = np.unique(pcdcolor[np.where(pcd[:,0] == x)], axis=0, return_counts=True)
            #print(colorcounts)
            #colors, colorcounts = np.unique(pcdcolor[np.where(pcd[:,0] == x)], axis=0, return_counts=True)
            for c,count in zip(colors,colorcounts):
                mem = tuple(c)
                #mem = roundcolor(tuple(c*255)) #3pcd stores normalized colors
                nodes = membership[mem]
                if len(nodes) > 1:
                    for n in nodes:
                        colorprop[int(n)] = colorprop[int(n)] + count/len(nodes)
                elif len(nodes) == 1:
                    colorprop[int(nodes[0])] = colorprop[int(nodes[0])] + count
                else:
                    print('Problem; no nodes')
            #normalize
            colorprop = colorprop/np.sum(colorcounts)
            #print(np.sum(colorprop))
            colormatrix.append(colorprop)
    return np.asarray(colormatrix)



def get_embedding_from_color_matrix(colormatrix,similarity):
    embedding = np.transpose(np.matmul(colormatrix,similarity))
    return embedding


    

def flipX(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,0] = -pcdpts[:,0]
    fac = np.min(pcdpts[:,0])
    #print(fac)
    pcdpts[:,0] = -fac + pcdpts[:,0]
    return pcdpts

def flipY(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,1] = -pcdpts[:,1]
    fac = np.min(pcdpts[:,1])
    #print(fac)
    pcdpts[:,1] = -fac + pcdpts[:,1]
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
 

def rotateForLayeringOption2WAug(pcd,aug):

    pcdpts = np.asarray(pcd.points)[:-2,:]
    ptscolor = np.asarray(pcd.colors)[:-2,:]
    if aug == 0:
        pts = pcdpts
    elif aug == 1:
        pts = flipX(pcdpts)
    elif aug == 2:
        pts = flipY(pcdpts)
    else:
        pts = flipX(flipY(pcdpts))
    
    newpcd = o3d.geometry.PointCloud()
    newpcd.points = o3d.utility.Vector3dVector(pts)
    newpcd.colors = o3d.utility.Vector3dVector(ptscolor)
    
    R45 = o3d.geometry.get_rotation_matrix_from_xyz([0,-np.pi/4,0])
    newpcd.rotate(R45)

    return newpcd 

def orientCamBottom(pcd):
    campos = np.asarray(pcd.points)[-1,:] 
    if campos[2] > 0:
        Rtemp = o3d.geometry.get_rotation_matrix_from_xyz([np.pi,0,0])
        pcd.rotate(Rtemp)
    return pcd

def roundcolor(colortuple):
    base = 5
    return tuple([ int(base * round(p/base)) for p in colortuple])

def roundpcdcolor(ptscolors):
    rounded = []
    for ct in ptscolors:
        temp = list(roundcolor(tuple(ct*255)))
        rounded.append(temp)
    
    return np.asarray(rounded)

version = 'v6'
membership = np.load('/home/smartslab/Desktop/Appearance/after_general/Mapper/node_membership_lookup_new2dlens_hue_saturation_withfake.npy',allow_pickle=True).item()
similarity = np.load('/home/smartslab/Desktop/Appearance/after_general/Mapper/similarity_'+version+'_new2dlens_hue_saturation_withfake.npy')
nnodes = np.shape(similarity)[0]

model_type = 'all'

    
if model_type == 'all':
    cam_a = [i for i in range(0,360,5)]
    cam_b = [i for i in range(0,185,5)]
    cam_a_remove = []
    cam_b_remove = [0,5,175,180]     
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))
            
            

object_list = os.listdir('./library/')
for oname in object_list[0:8]:
    data = {} 
    print(oname)
    maxlayers = 0
    instances = {}

    for bdeg in  cam_b_final:
        folder = str(bdeg)+'/0'
        print(folder)
        for file in cam_a_final:
            for aug in range(4):
                pcd = o3d.io.read_point_cloud('./library/'+oname+'/'+folder+'/'+'flatcolorpcdwcam/'+str(file)+'.pcd')
                
                trpcd = trXMinusCam(trYMinusCam(trZMinusCam(pcd)))
                rotatedpcd = orientCamBottom(trpcd)
                finaltrpcd = trXMinusCam(trYMinusCam(trZMinusCam(rotatedpcd)))

                rotatedpcd = rotateForLayeringOption2WAug(finaltrpcd,aug)
                finalpcd = trXMinusCam(trYMinusCam(trZMinusCam(rotatedpcd)))
                finalpcd.has_colors()
                
                pcdpts = np.asarray(finalpcd.points)
                ptscolors = np.asarray(finalpcd.colors)
                rounded = roundinXYZ(pcdpts) 
                zs = getZs(rounded)
                
                pis = {}
                for key,value in zs.items():
                    layer, layercolor = getColorLayer(ptscolors,rounded,zs,key)
                    
                    colormatrix = computeColorEmbeddingNo2DTranslation(layer, layercolor, membership,nnodes)
                    embed = get_embedding_from_color_matrix(colormatrix,similarity)

                    pis[key] = embed
                maxlayers = max(maxlayers,key+1)
                instances['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)] = pis
    data[oname] = (instances,maxlayers)
    np.save('./libembeds'+version+'/train1_library_allembeds_'+oname+'.npy',data)    
