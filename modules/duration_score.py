import numpy as np
import scipy.sparse as sps

def duration_all_items(ICM, feature = 2):
    
    duration_item_all = np.array(ICM[:,feature], dtype=int).ravel()
 
    # there are items with duration = 0. need to normalize
    duration_item_all_norm = np.add(duration_item_all, 1)
    
    return duration_item_all_norm

def calculate_duration_score(user_id, duration_item_all_norm, URM, ICM, feature = 2):

        # index of item of user
        rated_items = URM[user_id].indices
       
        # features selected 
        feature_user_items = np.array(ICM[rated_items, feature], dtype=int).squeeze()     
        features_user_items_norm = np.add(feature_user_items, 1)
        
        # zero score if no history
        if(features_user_items_norm.size == 0):
            return np.zeros(shape = URM.shape[1])

        # score of feature computing
        avg = features_user_items_norm.mean()

        # compute the score as: 1/sqr|(duration - avg)|
        abs_difference = np.abs(np.subtract(duration_item_all_norm,  avg))        
       
        # abs_difference with 1 are == avg. THE BEST MATCH if avg is int
        abs_difference[abs_difference == 0] = 1

        square_root = np.sqrt(abs_difference)
        reciprocal = np.reciprocal(square_root)

        # put to zero the items of the user which are inf
        np.put(reciprocal, rated_items, 0)

        # final score
        score = reciprocal
        return score


