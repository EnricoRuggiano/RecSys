from .recommender import Recommender
import modules.Compute_Similarity_Python as sim

import time as time
import numpy as np
import scipy.sparse as sps

class CFIB_recommender (Recommender):

    def fit(self, URM, topK = 100, shrink = 100):
        
        sim_obj = sim.Compute_Similarity_Python(URM, topK = topK, shrink = shrink, similarity = 'cosine')
        self.S = sim_obj.compute_similarity()
 
    def recommend_all(self, URM, test_set, at=10):
        
        submission = []
        S = self.S
        time_start = time_curr = time.time() 
       
        # Calculate matrix product: 
        # elements are sij where i is the item, j the user and
        # sij is the sum between the similarity score of item i with all the others.
        # the sum is done only on the coefficients of items with interactions.
        matrix_product = np.dot(S, URM.T)
        matrix_product_csc = sps.csc_matrix(matrix_product)
    
        # number of item interactions of the users
        # 50000 operations
        corated_items = [len(URM[user].indices) for user \
                in np.arange(URM.shape[0])]
        
        for user in test_set:
            
            #time
            if(time.time() - time_curr > 120):
                print("Time spent:\t {}sec Percentage of users seen:\t {:2.0f}%".format(\
                        (time_start - time.time()),(100 * (user / len(test_set)))))
                time_curr = time.time()
            
            # take a single column of product matrix. column_product has size item numbers.
            column_product = matrix_product_csc[:,user].toarray().squeeze()
            
            # all sij are < 1 and weighted by the occurency of the items
            column_norm = np.divide(column_product, corated_items[user])
            
            # put to zero all nan values; it happens only on testing
            column_norm_no_nan = np.where(np.isnan(column_norm), [0], column_norm) 
            
            # put to zero the items score already chosen by the user 
            mask_one_interaction = np.logical_not(URM[user].toarray().squeeze()) 
            column_ratings = np.multiply(column_norm_no_nan, mask_one_interaction)

            # sort by ratings and get the top ten items to recommend
            column_rank = np.argsort(column_ratings)[::-1][:10]
            submission.append([user, column_rank])
        
        return submission

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
          
