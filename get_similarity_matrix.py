#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani

"""

import heapq
import sys
import math
import numpy as np
import colour
import matplotlib.pyplot as plt
 

class Node:
    def __init__(self, node, cost):

        self.node = node
        self.cost = cost
 
    def __lt__(self, other):
        return self.cost < other.cost
 

def addEdge(adj, x, y, w):
    adj[x].append(Node(y, w))
    adj[y].append(Node(x, w))
 

def dijkstra(adj, src, n):
    pq = []
    settled = set()
 
    dist = [sys.maxsize] * (n+1)
    paths = [0] * (n+1)
 
    heapq.heappush(pq, Node(src, 0))
    dist[src] = 0
    paths[src] = 1
 
    while pq:
        u = heapq.heappop(pq)
        if (u.node, u.cost) in settled:
            continue
 
        settled.add((u.node, u.cost))
 
        for i in range(len(adj[u.node])):
            to = adj[u.node][i].node
            cost = adj[u.node][i].cost
 
            if (to, cost) in settled:
                continue
 
            if dist[to] > dist[u.node] + cost:
                dist[to] = dist[u.node] + cost
                paths[to] = paths[u.node]
                heapq.heappush(pq, Node(to, dist[to]))
            elif dist[to] == dist[u.node] + cost:
                paths[to] += paths[u.node]
 
    return dist, paths
 
def findShortestPaths(adj, s, n):
    dist, paths = dijkstra(adj, s, n)    
    return dist
        
        
def create_weighted_graph(graph,undirected):
    nnodes = len(undirected)
    adj = [[] for i in range(nnodes)]
    
    for key,value in undirected.items():
        i = int(key.split('_')[0][4:])
        for v in value:
            j = int(v.split('_')[0][4:])

            colori = np.mean(rgb[graph['nodes']['cube'+str(i)+'_cluster0']],axis=0)
            colorj = np.mean(rgb[graph['nodes']['cube'+str(j)+'_cluster0']],axis=0)
            dist = color_difference(rgb_to_lab(colori),rgb_to_lab(colorj))
            
            addEdge(adj,i,j,dist)
 
    return adj


def load_mapper(mapper):
    graph = np.load(mapper, allow_pickle=True).item()
    
    return graph

def convert_to_undirected(mapper):
    newgraph = {}
    for node,_ in mapper['nodes'].items():
        newgraph[node] = []
        
    for key,value in mapper['links'].items():
        for v in value:
            newgraph[key].append(v)
            newgraph[v].append(key)
            
    for key,value in newgraph.items():
        newgraph[key] = list(set(value))
    return newgraph



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


def get_similarity_matrix_v6(weighted,undirected):
    
    nnodes = len(undirected)
    
    distances = np.zeros((nnodes,nnodes))
    
    for i in range(nnodes):
        dist = findShortestPaths(weighted, i, nnodes-1)
        distances[i,:] = dist [0:22]
            
    mu = 1
    return 1/(1+(mu*distances))

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

mapperoutputfile = './lab_regions_new2dlens_hue_saturation_withfake.npy'
graph = load_mapper(mapperoutputfile)
undirected = convert_to_undirected(graph)
nnodes = len(undirected)
weighted = create_weighted_graph(graph,undirected)

sim = get_similarity_matrix_v6(weighted, undirected)
np.save('similarity_v6_new2dlens_hue_saturation_withfake.npy',sim)
