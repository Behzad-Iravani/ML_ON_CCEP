# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:06:42 2023

This script performs serach light with sliding window to charachterize ccep profile of 
corticothalamic thalamocortical corticocortical (OFC) and thalmothalamic 
@author: behira
"""
# import libraries 
import copy 
import matplotlib.pyplot as plt

from scipy.io import loadmat
import numpy as np
import mne
from mne.decoding import SlidingEstimator, Vectorizer

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import  StratifiedKFold
# from sklearn.metrics import balanced_accuracy
import sklearn.svm as svm

import pickle

class data:
    def __init__(self, path):
        info = mne.create_info(['chan'], 1000.0)
        l = preprocessing.LabelEncoder()
        dat = loadmat(path)
        self.tcs = np.expand_dims(dat["dat"], 1)
        self.label = [items[0][0] for items in dat["label"]]
        
        newlabel = []
        for items in self.label:
            # Split string on '.' 
            parts = items.split('.')
            if parts[0] == 'ANT' or parts[0] == 'MID' or parts[0] == 'PUL':
                if parts[1] == 'ORB' or parts[1] == 'vmPFC':
                    newlabel.append('Thalamocortical')
                elif parts[1] == 'ANT' or parts[1] == 'MID' or parts[1] == 'PUL':
                    newlabel.append('ThalamoThalamic')
            elif parts[0] == 'ORB' or parts[0] == 'vmPFC':
                if parts[1] == 'ORB' or parts[1] == 'vmPFC':
                    newlabel.append('CorticoCortical')
                elif parts[1] == 'ANT' or parts[1] == 'MID' or parts[1] == 'PUL':
                    newlabel.append('CorticoThalamic')

        self.l = l.fit(newlabel)
        events = np.concatenate((np.array([range(0, len(self.label))]),
                                 np.expand_dims(np.array(l.transform(newlabel)), axis=0),
                                 np.ones(shape=(1, len(self.label)))), axis=0).T
        # event_id= list(l.transform(self.label))
        self.epochs = mne.EpochsArray(self.tcs, info, events=events.astype(np.int64))
        
    def overlapping(self, data, window):

        newdata = [];
        for items in data:
            current = 0
            innerdata = []
            while current < (items.shape[-1] - window):
                innerdata.append(np.expand_dims(items[0, current:current + window], -1))
                current += window // 4
            newdata.append(np.expand_dims(np.concatenate(innerdata[1:len(innerdata) - 1], axis=1), -1))

        newdata[:] = np.expand_dims(np.concatenate(np.array(newdata), -1), -1)
        return newdata

    def applyCrossValidation(self, data, labels, classifier):
        window = 5
        CV_score_time = None
        data = self.overlapping(data, window)
        data = np.moveaxis(np.concatenate(data, axis=-1), 1, 0)
        data = np.moveaxis(data, 1, -1)

        #  SPLIT to train test and cross validate 
        skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=4)
    

        estimator = SlidingEstimator(classifier, scoring='balanced_accuracy',  verbose='DEBUG')
        
        # Lists to store estimators and scores
        estimators = []
        scores = []
        if np.isfinite(data).all() == True and np.isnan(data).any() == False:
            cv = 0
            for index_train, index_test in skf.split(data, labels):
                cv +=1
                if cv == 7 :
                    break
                X_train, X_test = data[index_train], data[index_test] 
                y_train, y_test = labels[index_train], labels[index_test]
                estimator.fit(X_train, y_train)
                estimators.append(copy.deepcopy(estimator))
                scores.append(np.expand_dims(estimator.score(X_test, y_test), axis=1))
        
            with open('CCEP_time_resolved_ML.pkl', 'wb') as file:        
                    pickle.dump(estimators, file)   
            with open('CCEP_time_resolved_scores.pkl', 'wb') as file:        
                    pickle.dump(scores, file) 

        else:
            print('Input contains NaN or infinity!')


    def CVscore(self):
        data_UN = self.epochs.get_data()
        labels_UN = self.epochs.events[:, -2]

        # remove nans 
        nanflag = np.isnan(data_UN).any(axis=2);
        rm = []
        for ix in range(data_UN.shape[0]):
            if nanflag[ix][0]:
                rm.append(ix)
        data_UN = np.delete(data_UN, rm, axis=0)
        labels_UN = np.delete(labels_UN, rm, axis=0)
        # -----------------------------
        CV_score_time_UN = []

        clf = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(kernel='linear'))
        CV_score_time_UN.append(self.applyCrossValidation(data_UN, labels_UN, clf))

if __name__ == "__main__":
    # instanitate the data class
    dat = data("F:\CCEP\script\ML_python_data.mat")
    # Epoching
    dat.CVscore()
