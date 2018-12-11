import numpy as np
import scipy.sparse as sps

import multiprocessing
from multiprocessing import Pool
from functools import partial
import threading

import modules.importer as imp

#URM = imp.create_URM()
#ICM = imp.create_ICM()

# return a matrix feature x item with 0 or 1
def feature_matrix(ICM, feature):
    
    FEATURE = np.zeros(((np.int32(np.max(ICM[:, feature]))) + 1, ICM.shape[0]))
    for track in np.arange(ICM.shape[0]):
        FEATURE[np.int32(ICM[track, feature]), track] = 1
    
    FEATURE_lil = sps.lil_matrix(FEATURE)
    return FEATURE_lil

def feature_score(user, feature, FEATURE, URM, ICM): 
    RESULT = FEATURE.copy()
    index = 0
    
    rated_items = URM[user].indices
    
    feature_user_items = np.array(ICM[rated_items, feature], dtype=int).squeeze()    
    feature_user, feature_occurency = np.unique(feature_user_items, return_counts = True)
    frequency_items = np.divide(feature_occurency, feature_user_items.shape)
    
    # iterate over rows 
     
    for feature_index in feature_user:
        RESULT[feature_index] = np.multiply(RESULT[feature_index], frequency_items[index])
        index = index + 1

    RESULT[RESULT == 1] = 0
    RESULT[:, rated_items] = 0

    rank = np.array(np.sum(RESULT, axis = 0)[0]).squeeze()
    return rank

def score_all(feature, target_playlist, URM, ICM):
    
    ICM_dense = ICM.todense()
    FEATURE = feature_matrix(ICM_dense, feature)
   
    pool = Pool(processes=multiprocessing.cpu_count())
    _feature_score = partial(feature_score, feature=feature, FEATURE = FEATURE, URM=URM, ICM=ICM_dense)
    
    res = pool.map(_feature_score, target_playlist)
    return res
