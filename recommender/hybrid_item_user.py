from .recommender import Recommender
import modules.Compute_Similarity_Python as sim

import time as time
import numpy as np
import scipy.sparse as sps

class CFHY_recommender (Recommender):

    def fit(self, URM, topK = 100, shrink = 100):
        
        sim_obj_user = sim.Compute_Similarity_Python(URM.T, topK = topK, shrink = shrink, similarity = 'cosine')
        sim_obj_item = sim.Compute_Similarity_Python(URM, topK=topK, shrink= shrink, similarity='cosine')
        
        self.S_user = sim_obj_user.compute_similarity()
        self.S_item = sim_obj_item.compute_similarity()
        
        self.FEATURE = self.feature_matrix(self.ICM.todense(), 0) # artist feature 

    def user_based_score(self, row_product, corated_users, user, URM):
        # all sij are < 1 and weighted by the occurency of the items
        row_norm = np.divide(row_product, corated_users)
            
        # put to zero all nan values; it happens only on testing
        row_norm_no_nan = np.where(np.isnan(row_norm), [0], row_norm) 
            
        # put to zero the items score already chosen by the user 
        mask_one_interaction = np.logical_not(URM[user].toarray().squeeze())
        row_ratings = np.multiply(row_norm_no_nan, mask_one_interaction)
       
        return row_ratings     
            
    def item_based_score(self, column_product, corated_items, user, URM):
        # all sij are < 1 and weighted by the occurency of the items
        column_norm = np.divide(column_product, corated_items[user])
            
        # put to zero all nan values; it happens only on testing
        column_norm_no_nan = np.where(np.isnan(column_norm), [0], column_norm) 
            
        # put to zero the items score already chosen by the user 
        mask_one_interaction = np.logical_not(URM[user].toarray().squeeze()) 
        column_ratings = np.multiply(column_norm_no_nan, mask_one_interaction)

        return column_ratings
		
    def feature_matrix(self, ICM, feature):
        
        FEATURE = np.zeros(((np.int32(np.max(ICM[:, feature]))) + 1, ICM.shape[0]))
        for track in np.arange(ICM.shape[0]):
            FEATURE[np.int32(ICM[track, feature]), track] = 1
            
        FEATURE_csr = sps.csr_matrix(FEATURE)
        return FEATURE_csr

    def calculate_artist_score(self, user, feature, FEATURE, URM, ICM): 
        RESULT = sps.lil_matrix(FEATURE, copy = True)
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

    def combine(self, user_score, item_score, duration_score, weights, square = True):
               
        if(square):
            user_score_weigthed = np.multiply(np.power(2, user_score), weights[0])
            item_score_weigthed = np.multiply(np.power(2, item_score), weights[1])
            duration_score_weigthed = np.multiply(np.power(2, duration_score), weights[2])
            
            sum_score = np.add(user_score_weigthed, item_score_weigthed, duration_score_weigthed)
            total_score = np.sqrt(sum_score)
        else:
            user_score_weigthed = np.multiply(user_score_weigthed, weights[0])
            item_score_weigthed = np.multiply(item_score_weigthed, weights[1])
            duration_score_weigthed = np.multiply(duration_score_weigthed, weights[2]) 
            
            sum_score = np.add(user_score, item_score, duration_score_weigthed)
            total_score = sum_score
        
        return total_score
        
    def recommend_all(self, URM, test_set, at=10):
        
        submission = []
        S_user = self.S_user
        S_item = self.S_item
        URM_csc = sps.csc_matrix(URM)
        time_start = time_curr = time.time() 
       
        # item content matrix
        ICM = self.ICM.todense()
        features_item_all = np.array([np.array(ICM[item]).squeeze() for item in np.arange(URM.shape[1])]) 
        features_item_all_norm = np.add(features_item_all[2], 1)
       
        # Calculate matrix product: 
        # elements are sij where i is the user, j item and
        # sij the sum of all similarity of users with interaction 1
        matrix_product_item = np.dot(S_item, URM.T)
        matrix_product_user = np.dot(S_user, URM)

        matrix_product_csc = sps.csc_matrix(matrix_product_item)
        
        # number of users with interaction of items
        # 20000 operations
        corated_users = [len(URM_csc[:, item].indices) for item \
                in np.arange(URM.shape[1])]
        
        # number of item interactions of the users
        # 50000 operations
        corated_items = [len(URM[user].indices) for user \
                in np.arange(URM.shape[0])]
        
        for user in test_set:
            
            #time
            if(time.time() - time_curr > 120):
                print("Time spent:\t {} min Percentage of users seen:\t {:2.0f}%".format(\
                        (time.time() - time_start / 60),(100 * (user / len(test_set)))))
                time_curr = time.time()
            
            # take a single row of product matrix. matrix_product[user] is an np.matrix
            row_product = matrix_product_user[user].toarray().squeeze()
            # take a single column of product matrix. column_product has size item numbers.
            column_product = matrix_product_csc[:,user].toarray().squeeze()
             
            # user based score of user vector of items
            user_score = self.user_based_score(row_product, corated_users, user, URM) 
            # item based score of user vector of items
            item_score = self.item_based_score(column_product, corated_items, user, URM)
			# duration score
#           duration_score = self.calculate_duration_score(user, features_item_all_norm[2], URM, ICM)
			# artist score
            artist_score = self.calculate_artist_score(user, 0, self.FEATURE, URM, ICM)
	    # album score
	    #artist_score = self.calculate_artist_score(user, features_item_all_norm[0], URM, ICM)
			
            # combine the score with a weigthed sum
            total_score = self.combine(user_score, item_score, artist_score, [0.25, 0.75, 2])
           
            # sort and get the top ten items with best score
            submission_score = np.argsort(total_score)[::-1][:at]
            submission.append([user, submission_score])
        
        return submission

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
 
