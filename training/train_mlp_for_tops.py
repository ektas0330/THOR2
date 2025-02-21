#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""

import cv2
import numpy as np
import fnmatch,os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import copy 
import pickle
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
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

def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 100:
        lr *= 1e-1
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def getMLPInput(trainingData,numLayers):
    idxes = range(0,numLayers)
    datainput = trainingData[0]
    if len(idxes) == 1:
        return datainput
    else:
        for i in idxes[1:]:
            datainput = np.concatenate((datainput,trainingData[i]),axis=1)
        return datainput
        

def getLthPIs(data,l):
    pi = []
    for key,value in data.items():
        for k,v in value[0].items():
            if l in v:
                pi.append(np.reshape(value[0][k][l],(1024,)))
            else:
                pi.append(np.ones((1024,)))           
    return pi


def getLabels(data):
    labels = []
    for key,value in data.items():
        for k,v in value[0].items():
            labels.append(object_list.index(key))
    return labels


def main(data_path,random_state):
    model_type = 'all'
    cam_a = [i for i in range(0,360,5)]
    cam_b = [i for i in range(0,185,5)]
    cam_a_remove = []
    cam_b_remove = [0,5,175,180]     
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))

    data = {}
    
    object_list = os.listdir(data_path)
    
    seq = 'tops_all_layers'

    if not os.path.exists('./mlp_models/'):
        os.makedirs('./mlp_models/')
    if not os.path.exists('./mlp_models/'+seq):
        os.makedirs('./mlp_models/'+seq)


    for oname in object_list:
        alldata = np.load(np.load('./libpis/allpis_'+oname+'.npy',allow_pickle=True).item()
        maxlayers = 0
        instances = {}
        
        bdeglist = cam_b_final

        for bdeg in  bdeglist:
            folder = str(bdeg)+'/0'
            for file in cam_a_final:
                for aug in range(4):
                    instances['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)] = alldata[oname][0]['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)]
                
        data[oname] = (instances,alldata[oname][1])

        overallMaxLayers = 0
        for key,value in data.items():
            overallMaxLayers = max(value[1],overallMaxLayers)

        trainingData = {}
        for l in range(overallMaxLayers):
            trainingData[l] = np.asarray(getLthPIs(data,l))

        labels = getLabels(data)

        l = overallMaxLayers

        xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getMLPInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=2019)

        train_encoded_labels = to_categorical(ytrain)
        val_encoded_labels = to_categorical(ytest)

        model = classifier_mlp_softmax(len(object_list),l)
        model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr_schedule(0)),metrics=['accuracy'])

        if not os.path.exists(os.path.dirname('./mlp_models/'+seq+ '/mlp_'+model_type+'_' + str(l)+'layers.hdf5')):
            os.makedirs(os.path.dirname('./mlp_models/'+seq+ '/mlp_'+model_type+'_' + str(l)+'layers.hdf5'))
        if os.path.isfile('./mlp_models/'+seq+ '/mlp_'+model_type+'_' + str(l)+'layers.hdf5'):
            os.remove('./mlp_models/'+seq+ '/mlp_'+model_type+'_' + str(l)+'layers.hdf5')

        filepath = './mlp_models/'+seq+ '/mlp_'+model_type+'_' + str(l)+'layers.hdf5'

        checkpoint = ModelCheckpoint(filepath=filepath,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True)

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                        cooldown=0,
                                        patience=10,
                                        min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler]


        history = model.fit(xtrain, train_encoded_labels,
                    batch_size=16,
                epochs=100,
                    validation_data=(xtest, val_encoded_labels),
                    verbose=2,
                    shuffle=True,
                    callbacks=callbacks)

        (loss, accuracy) = model.evaluate(xtest, val_encoded_labels,batch_size=64,verbose=1)
        print('[INFO] accuracy: {:.2f}%'.format(accuracy * 100))
        model.save_weights('./mlp_models/'+seq+ '/mlp_'+model_type+'_' + str(l)+'layers.hdf5', overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--random_state')
    args = parser.parse_args()
    main(args.data_path,args.random_state)
