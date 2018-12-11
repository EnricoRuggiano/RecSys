from .recommender import Recommender
import modules.Compute_Similarity_Python as sim

import time as time
import numpy as np
import scipy.sparse as sps
import sklearn.utils.extmath as skl

class CFHYS_recommender (Recommender):

    def fit(self, URM, topK = 100, shrink = 100):
        
        sim_obj_item = sim.Compute_Similarity_Python(URM, topK=topK, shrink= shrink, similarity='cosine')
        self.S_item = sim_obj_item.compute_similarity()
        
        self.u, self.sigma, self.v = skl.randomized_svd(URM, n_components = 100)
        self.s_Vt = sps.diags(self.sigma) * self.v

  
    def svd_based_score(self, user, URM):
        # calculate ranks
        rank = self.u[user,:].dot(self.s_Vt)
        
        #put zero the items of users 
        np.put(rank, URM[user].indices, 0)
        return rank     
            
    def item_based_score(self, column_product, corated_items, user, URM):
        # all sij are < 1 and weighted by the occurency of the items
        column_norm = np.divide(column_product, corated_items[user])
            
        # put to zero all nan values; it happens only on testing
        column_norm_no_nan = np.where(np.isnan(column_norm), [0], column_norm) 
            
        # put to zero the items score already chosen by the user 
        mask_one_interaction = np.logical_not(URM[user].toarray().squeeze()) 
        column_ratings = np.multiply(column_norm_no_nan, mask_one_interaction)

        return column_ratings
		
    def combine(self, item_score, svd_score, weights, square = True):
               
        if(square):
            item_score_weigthed = np.multiply(np.power(2, item_score), weights[0])
            svd_score_weigthed = np.multiply(np.power(2, svd_score), weights[1])
            
            sum_score = np.add(item_score_weigthed, svd_score_weigthed)
            total_score = np.sqrt(sum_score)
        else:
            item_score_weigthed = np.multiply(item_score_weigthed, weights[0])
            svd_score_weighted = np.multiply(svd_score_weigthed, weights[1]) 
            
            sum_score = np.add(item_score, svd_score_weigthed)
            total_score = sum_score
        
        return total_score
        
    def recommend_all(self, URM, test_set, at=10):
        
        submission = []
        S_item = self.S_item
        URM_csc = sps.csc_matrix(URM)
        time_start = time_curr = time.time() 
       
        # item content matrix
        #features_item_all = np.array([np.array(ICM[item]).squeeze() for item in np.arange(URM.shape[1])]) 
        #features_item_all_norm = np.add(features_item_all[2], 1)
       
        # Calculate matrix product: 
        # elements are sij where i is the user, j item and
        # sij the sum of all similarity of users with interaction 1
        matrix_product_item = np.dot(S_item, URM.T)
        
        matrix_product_csc = sps.csc_matrix(matrix_product_item)
        
        # number of item interactions of the users
        # 50000 operations
        corated_items = [len(URM[user].indices) for user \
                in np.arange(URM.shape[0])]
        
        for user in test_set:
            
            #time
            if(time.time() - time_curr > 120):
                print("Time spent:\t {} min Percentage of users seen:\t {:2.0f}%".format(\
                        (time.time() - time_start),(100 * (user / len(test_set)))))
                time_curr = time.time()
            
            # take a single column of product matrix. column_product has size item numbers.
            column_product = matrix_product_csc[:,user].toarray().squeeze()
             
            # item based score of user vector of items
            item_score = self.item_based_score(column_product, corated_items, user, URM)
	
            # svd bsed score
            svd_score = self.svd_based_score(user, URM)
	   	
            # combine the score with a weigthed sum
            total_score = self.combine(item_score, svd_score, [0.9, 0.1])
           
            # sort and get the top ten items with best score
            submission_score = np.argsort(total_score)[::-1][:at]
            submission.append([user, submission_score])
        
        return submission

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
 
