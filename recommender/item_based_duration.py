from .recommender import Recommender
import modules.Compute_Similarity_Python as sim

import time as time
import numpy as np
import scipy.sparse as sps

class CFIBD_recommender (Recommender):

    def fit(self, URM, topK = 100, shrink = 100):
        
        sim_obj = sim.Compute_Similarity_Python(URM, topK = topK, shrink = shrink, similarity = 'cosine')
        self.S = sim_obj.compute_similarity()
    
    def combine(self, item_score, duration_score,  weights=[0.99, 0.01], square = True):
               
        if(square):
            item_score_weigthed = np.multiply(np.power(2, item_score), weights[0])
            duration_score_weigthed = np.multiply(np.power(2, duration_score), weights[1])

            sum_score = np.add(item_score_weigthed, duration_score_weigthed)
            total_score = np.sqrt(sum_score)
        else:
            item_score_weigthed = np.multiply(item_score, weights[0]) 
            duration_score_weigthed = np.multiply(duration_score, weights[1])
            
            sum_score = np.add(item_score_weigthed, duration_score_weigthed)
            total_score = sum_score
        
        return total_score
     
    def calculate_duration_score(self, user, original_features_item_all_norm, URM, ICM):

        # index of item of user
        items_of_user = URM[user].indices
       
        # features selected
        features_items_of_user = np.array([np.array(ICM[item]).squeeze()[2] for item in items_of_user]) 
        features_items_of_user_norm = np.add(features_items_of_user, 1)
        features_item_all_norm = original_features_item_all_norm

        # zero score if no history
        if(features_items_of_user_norm.size == 0):
            return np.zeros(shape = URM.shape[1])

        # score of feature computing
        avg_of_durations = features_items_of_user_norm.mean()
        mask_avg = np.full(URM.shape[1], avg_of_durations)

        # compute the score as: 1/sqr|(duration - avg)|
        abs_difference = np.abs(np.subtract(features_item_all_norm, mask_avg))        
        abs_difference_norm = np.add(abs_difference, 1)
        square_root = np.sqrt(abs_difference_norm)
        reciprocal = np.reciprocal(square_root)

        # put to zero the items of the user which are inf
        np.put(reciprocal, items_of_user, 0)

        # final score
        score = reciprocal
        return score

    def recommend_all(self, URM, test_set, at=10):
        
        submission = []
        S = self.S
        time_start = time_curr = time.time() 

        ICM = self.ICM.todense()
        features_item_all = np.array([np.array(ICM[item]).squeeze()[2] for item in np.arange(URM.shape[1])]) 
        features_item_all_norm = np.add(features_item_all, 1)

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
                        (time.time() - time_start),(100 * (user / len(test_set)))))
                time_curr = time.time()
            
            # take a single column of product matrix. column_product has size item numbers.
            column_product = matrix_product_csc[:,user].toarray().squeeze()
            
            # all sij are < 1 and weighted by the occurency of the items
            column_norm = np.divide(column_product, corated_items[user])
            
            # put to zero all nan values; it happens only on testing
            column_norm_no_nan = np.where(np.isnan(column_norm), [0], column_norm) 
            
            # put to zero the items score already chosen by the user 
            mask_one_interaction = np.logical_not(URM[user].toarray().squeeze()) 
            item_score = np.multiply(column_norm_no_nan, mask_one_interaction)

            # feature score
            feature_score = self.calculate_duration_score(user, features_item_all_norm, URM, ICM)

            # calculate final score combining them
            final_score = self.combine(item_score, feature_score)

            # sort by ratings and get the top ten items to recommend
            rank = np.argsort(final_score)[::-1][:10]
            submission.append([user, rank])
        
        return submission

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
          
