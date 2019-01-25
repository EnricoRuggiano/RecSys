import numpy as np
import scipy.sparse as sps
import modules.sequential_score as sequential

def item_score(user_id, URM, S, sequential_list):
    weigths = sequential.sequential_score(sequential_list, URM, user_id)
    S_weigthed = np.dot(S, sps.diags(weigths))
    score = np.dot(URM[user_id], S_weigthed).toarray().squeeze()
    
    # remove seen items
    np.put(score, URM[user_id].indices, [0])
    return score
