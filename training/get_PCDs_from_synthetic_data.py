#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""
import open3d as o3d
import csv,os
import numpy as np
import ast
import argparse

def main(data_path):
    object_list = os.listdir(data_path)
    
    for obj_name in object_list:
        print(obj_name)
        for cam_b_deg in range(0,185,5):
            folder = str(cam_b_deg)+'/0'
            if not os.path.exists(data_path+obj_name+'/'+folder+'/'+'pcd/'):
                os.makedirs(data_path+obj_name+'/'+folder+'/'+'pcd/')
                
            intrinsics = {}
            #extrinsics = {}
            
            with open(data_path+obj_name+'/'+folder+'/'+obj_name+'.csv', mode='r') as inp:
                reader = csv.reader(inp)
                intrinsics = {rows[0]:np.reshape(ast.literal_eval(rows[1]),(3,3)) for rows in reader}
            
            
            for k,v in intrinsics.items():
                f = 1099 #fov40
                depth_img = np.squeeze(np.load(data_path+obj_name+'/'+folder+'/'+'depth/'+os.path.splitext(k)[0]+'.npy'))
                rgb_img = cv2.imread(data_path + obj_name+'/'+folder+'/'+'rgb/'+os.path.splitext(k)[0]+'.png')[...,::-1].copy()
                p2d_idx = np.where(depth_img>-1)
                us = p2d_idx[0]
                vs = p2d_idx[1]
                values = np.squeeze(depth_img[us,vs])
                cvalues = np.squeeze(rgb_img[us,vs])/255
                p2d_value = np.vstack((us,vs,values)).T
                p3d = p2d_value.copy()
                p3d[:,0] = -(p3d[:,0]-300)/f#/p3d[:,2]
                p3d[:,1] = (p3d[:,1]-400)/f#/p3d[:,2]
                objpcd = p3d[np.where(p3d[:,2]< 1.0)]
                objpcdcolor = cvalues[np.where(p3d[:,2]< 1.0)]
                pcl = o3d.geometry.PointCloud()
                pcl.points = o3d.utility.Vector3dVector(objpcd)
                pcl.colors = o3d.utility.Vector3dVector(objpcdcolor)
                o3d.io.write_point_cloud(data_path+obj_name+'/'+folder+'/'+'pcd/'+os.path.splitext(k)[0]+'.pcd', pcl)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()
    main(args.data_path)