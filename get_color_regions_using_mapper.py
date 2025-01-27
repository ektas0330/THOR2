#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""

import numpy as np
import math
import kmapper as km
import sklearn
import matplotlib.pyplot as plt
import colour
import copy

def rgb_to_lab(rgb):
    illuminant = np.array([0.31271, 0.32902]) #D65 std 2 degree observer
    xyz = colour.sRGB_to_XYZ(rgb, illuminant)  
    
    lab = colour.XYZ_to_Lab(xyz,illuminant)    
    return lab

def lab_to_rgb(lab):
    illuminant = np.array([0.31271, 0.32902]) #D65 std 2 degree observer
    xyz = colour.Lab_to_XYZ(lab,illuminant)  
    rgb = colour.XYZ_to_sRGB(xyz,illuminant)
    return rgb

def show_color(rgbcolor):
    assert(rgbcolor[0] < 1)
    colorfig = np.zeros((5,5,3))
    colorfig[:,:,0] = rgbcolor[0]*np.ones((5,5))
    colorfig[:,:,1] = rgbcolor[1]*np.ones((5,5))
    colorfig[:,:,2] = rgbcolor[2]*np.ones((5,5))
    plt.figure()
    plt.imshow(colorfig)
    

def color_difference(x,y): #from 2020 fairchild paper
    return abs(x[0] - y[0]) + math.sqrt((x[1] - y[1])**2 + (x[2] - y[2])**2   )


r_ = np.arange(0,256,5)
g_= np.arange(0,256,5)
b_ = np.arange(0,256,5)


rgb = []
for i in r_:
    for j in g_:
        for k in b_:
            rgb.append([i,j,k])
            
rgb = np.asarray(rgb)/255
lab = rgb_to_lab(rgb)


data = lab

def get_projection(data):
    projected = []
    for i in range(len(data)):
        curr = data[i,:]
        
        projected.append([np.sqrt(curr[2]**2 + curr[1]**2 ) ,22.5  + (180/np.pi) *np.arctan2(curr[2], curr[1])])
    return np.asarray(projected)

# Initialize
mapper = km.KeplerMapper(verbose=2)

projected_data = get_projection(data)

n_cubes = [3,8]
cover=km.Cover(n_cubes=n_cubes, perc_overlap=[0.1,0.25])


clusterer= sklearn.cluster.DBSCAN(eps=7, min_samples=6, metric=color_difference)

graph = mapper.map(projected_data, data, cover=cover,clusterer = clusterer)

# Visualize it
mapper.visualize(graph, path_html="lab_regions_new2dlens_hue_saturation.html",
                  title="lab_regions_new2dlens_hue_saturation", color_values = projected_data[:,0], color_function_name = ['L'])

np.save('lab_regions_new2dlens_hue_saturation.npy',graph)


mygraph = copy.deepcopy(graph)
mygraph['links']['cube0_cluster0'].append('cube7_cluster0')
mygraph['links']['cube8_cluster0'].append('cube15_cluster0')

mem18 = mygraph['nodes']['cube18_cluster0']
mem19 = mygraph['nodes']['cube19_cluster0']

mygraph['nodes']['cube18_cluster0'] = mem19 + list(set(mem18) - set(mem19))
mygraph['nodes']['cube19_cluster0'] = mygraph['nodes']['cube19_cluster1']
del mygraph['nodes']['cube19_cluster1']

mygraph['links']['cube12_cluster0'] = ['cube13_cluster0', 'cube17_cluster0', 'cube18_cluster0']
mygraph['links']['cube13_cluster0'] = ['cube14_cluster0', 'cube18_cluster0', 'cube19_cluster0', 'cube20_cluster0']
mygraph['links']['cube14_cluster0'] = ['cube15_cluster0', 'cube19_cluster0', 'cube20_cluster0', 'cube21_cluster0']
del mygraph['links']['cube19_cluster1']
mygraph['links']['cube19_cluster0'] = ['cube20_cluster0']

mapper.visualize(mygraph, path_html="lab_regions_new2dlens_hue_saturation_withfake.html",
                  title="lab_regions_new2dlens_hue_saturation_withfake", color_values = projected_data[:,0], color_function_name = ['L'])

np.save('lab_regions_new2dlens_hue_saturation_withfake.npy',mygraph)
