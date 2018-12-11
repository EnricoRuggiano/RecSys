from .recommender import Recommender
import modules.Compute_Similarity_Python as sim

import time as time
import numpy as np
import scipy.sparse as sps

class CFUB_recommender (Recommender):

    def fit(self, URM, topK = 200, shrink = 500):
        
        sim_obj = sim.Compute_Similarity_Python(URM.T, topK = topK, shrink = shrink, similarity = 'cosine')
        self.S = sim_obj.compute_similarity()
 
    def recommend_all(self, URM, test_set, at=10):
        
        submission = []
        S = self.S
        URM_csc = sps.csc_matrix(URM)
        time_start = time_curr = time.time() 
       
       # Calculate matrix product: 
        # elements are sij where i is the user, j item and
        # sij the sum of all similarity of users with interaction 1
        matrix_product = np.dot(S, URM)

        # number of users with interaction of items
        # 20000 operations
        corated_users = [len(URM_csc[:, item].indices) for item \
                in np.arange(URM.shape[1])]
        
        for user in test_set:
            
            #time
            if(time.time() - time_curr > 120):
                print("Time spent:\t {}sec Percentage of users seen:\t {:2.0f}%".format(\
                        (time_start - time.time()),(100 * (user / len(test_set)))))
                time_curr = time.time()
            
            # take a single row of product matrix. matrix_product[user] is an np.matrix
            row_product = matrix_product[user].toarray().squeeze()
            
            # all sij are < 1 and weighted by the occurency of the items
            row_norm = np.divide(row_product, corated_users)
            
            # put to zero all nan values; it happens only on testing
            row_norm_no_nan = np.where(np.isnan(row_norm), [0], row_norm) 
            
            # put to zero the items score already chosen by the user 
            mask_one_interaction = np.logical_not(URM[user].toarray().squeeze())
            row_ratings = np.multiply(row_norm_no_nan, mask_one_interaction)
            
            # sort by ratings and get the top ten items to recommend
            row_rank = np.argsort(row_ratings)[::-1][:10]
            submission.append([user, row_rank])
        
        return submission

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
   
