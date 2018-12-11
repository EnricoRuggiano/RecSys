from .recommender import Recommender
import modules.Compute_Similarity_Python as sim
import modules.artist_score_sps as art
import sklearn.utils.extmath as skl

import time as time
import scipy.sparse as sps
import numpy as np
from .Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from .default.Base.Evaluation.Evaluator import SequentialEvaluator

class SLIM_BPR(Recommender):
    
    def __init__(self, ICM, URM, URM_train, URM_test, target_playlists):
        
        Recommender.__init__(self,ICM, URM, URM_train, URM_test, target_playlists) 

    def fit(self, URM, artist = False):

        self.URM_setted = URM

        #sim_obj_user = sim.Compute_Similarity_Python(URM.T, topK = 100, shrink = 100, similarity = 'cosine')
        sim_obj_item = sim.Compute_Similarity_Python(URM, topK=100, shrink= 100, similarity='cosine')
        
        #self.S_user = sim_obj_user.compute_similarity()
        self.S_item = sim_obj_item.compute_similarity()
     
        # svd
        #self.u, self.sigma, self.v = skl.randomized_svd(URM, n_components = 100)
        #self.s_Vt = sps.diags(self.sigma) * self.v

        matrix_product_item = np.dot(self.S_item, URM.T)
        self.matrix_product_csc = sps.csc_matrix(matrix_product_item)
        
        #self.matrix_product_user = np.dot(self.S_user, URM)
       
        #URM_csc = sps.csc_matrix(URM)
        
        # number of users with interaction of items
        #self.corated_users = [len(URM_csc[:, item].indices) for item \
        #        in np.arange(URM.shape[1])]
 
        # number of item interactions of the users
        self.corated_items = [len(URM[user].indices) for user \
                in np.arange(URM.shape[0])]
        
 
        self.recommender = SLIM_BPR_Cython(URM, positive_threshold=1, recompile_cython = False, sparse_weights = True, sgd_mode='adagrad') 
        self.recommender.fit(epochs = 3,i sgd_mode='adagrad' , learning_rate=0.05, lambda_i = 0.1, lambda_j = 0.005)
        if(artist):
            self.ICM = self.ICM.todense()
            self.FEATURE = art.feature_matrix(self.ICM, 0)
            #            time_start = time.time()
#            self.artist_score = art.score_all(0, np.arange(self.URM_test.shape[0]), self.URM_train, self.ICM)

#            print("COMPLETED ARTIST SCORE:\t {} min".format(time_time()-time_start))

    # merge two score keeping the best value score at their items
    def merge(self, score_one, score_two, normalize = True):
        if(normalize):
            score_one = self.normalize(score_one) 
            score_two = self.normalize(score_two)
        score_merge = np.where(score_one > score_two, score_one, score_two)
        return score_merge

    def combine(self, item_score, svd_score, weights, square = True, normalize = True):
        
        if(normalize):
            item_score = self.normalize(item_score) 
            svd_score  = self.normalize(svd_score)
        if(square):
            item_score_weigthed = np.multiply(np.power(2, item_score), weights[0])
            svd_score_weigthed = np.multiply(np.power(2, svd_score), weights[1])
            
            sum_score = np.add(item_score_weigthed, svd_score_weigthed)
            total_score = np.sqrt(sum_score)
        else:
            item_score_weigthed = np.multiply(item_score, weights[0])
            svd_score_weighted = np.multiply(svd_score, weights[1]) 
            
            sum_score = np.add(item_score_weigthed, svd_score_weighted)
            total_score = sum_score
        
        return total_score

    def normalize(self, score):
        score_bias = np.abs(np.min(score))
        score_norm = np.linalg.norm(np.add(score, score_bias))
        score_normalized = np.divide(np.add(score, score_bias), score_norm)
        return score_normalized

    def svd_based_score(self, user, URM):
        # calculate ranks
        score = self.u[user,:].dot(self.s_Vt)
        
        #put zero the items of users 
        np.put(score, URM[user].indices, -np.inf)
        return score     
    
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
		
     
    def recommend_all(self):

        with open(self.output_path, 'w') as f: 
            print("playlist_id,track_ids", end='\n', file=f)
            for user_id in self.target_playlists:
                rank = self.recommend(user_id)
                print(user_id, end=',', file = f)
                print(*rank, sep=' ', end='\n', file = f)

    def recommend(self, user_id):

        # take a single column of product matrix. column_product has size item numbers.
        column_product = self.matrix_product_csc[:,user_id].toarray().squeeze()
        #row_product = self.matrix_product_user[user_id].toarray().squeeze()
             
        # item based score of user vector of items
        item_score = self.item_based_score(column_product, self.corated_items, user_id, self.URM_setted)	
        #user_score = self.user_based_score(row_product, self.corated_users, user_id, self.URM_setted) 
            
        # bpr score
        bpr_score = self.recommender.recommend(user_id, at=10, exclude_seen=True)
        #bpr_score = self.rec_based_score(user_id, self.URM_setted)
        bpr_not_inf = np.where(np.isinf(bpr_score), np.nan, bpr_score)
        bpr_not_nan = np.where(np.isnan(bpr_not_inf), np.nanmin(bpr_not_inf), bpr_not_inf) # assign the min score to already saw items
        bpr_score = bpr_not_nan
        
        #artist_score
#        artist_score = art.feature_score(user_id, 0, self.FEATURE, self.URM_setted, self.ICM)

        submission = self.combine(item_score, bpr_score, [1, 1], square=False, normalize=True)
        #submission = self.merge(item_score, bpr_score, normalize=True)

        score = np.argsort(submission)[::-1][:10]
        
        return score
