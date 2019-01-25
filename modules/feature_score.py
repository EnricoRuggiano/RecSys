import numpy as np
import scipy.sparse as sps

def calculate_feature_matrix(feature, URM, ICM):
    # items x feature sparse matrix
    indices = np.array(ICM[:,feature], dtype=int).ravel()
        
    n_items = URM.shape[1]
    n_feature = np.max(indices) + 1

    data = np.full(n_items, 1)
    indptr = np.arange(n_items + 1)
        
    f_matrix = sps.csc_matrix((data, indices, indptr), shape = (n_feature, n_items), dtype=np.float)
        
    f = f_matrix.tocsr()

    return f

def feature_score(user, feature, feature_matrix, URM, ICM, most_popular=True): 
    f = feature_matrix.copy()
    
    rated_items = URM[user].indices
    if (rated_items.size == 0):
        rank = np.zeros(URM.shape[1])
        return rank
    
    feature_user_items = np.array(ICM[rated_items, feature], dtype=int).squeeze()    
    feature_user, frequency_items = np.unique(feature_user_items, return_counts = True)
    #frequency_items = np.divide(feature_occurency, feature_user_items.shape)
    
    if (frequency_items.size == 0):
        rank = np.zeros(URM.shape[1])
        return rank
    
    for feature in feature_user:
        index = np.argwhere(feature_user==feature).squeeze()
        
        if(frequency_items[index] > 1):
            raw_start = feature
            raw_end = feature + 1
            f[raw_start:raw_end] = np.multiply(f[raw_start:raw_end], frequency_items[index])

    rank = np.array(np.sum(f, axis = 0)[0]).squeeze()

    rank[rated_items] = 0
    mask = np.in1d(np.arange(URM.shape[1]), f[feature_user].indices)
    rank = np.multiply(rank, mask)
    
    if(most_popular):
        popular_tracks = np.array(URM.sum(axis = 0)).squeeze()
        weigths = np.log10(popular_tracks)
        score = np.multiply(rank, weigths)

        # eliminate -np.inf
        indices = np.array(np.argwhere(weigths == -np.inf)).squeeze()
        np.put(score, indices, rank[indices])
        rank = score
    #rank[rank == 1] = 0

   # print("USER:\t {}".format(user))
    return rank
