from .recommender import Recommender
from .default.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet 
from .default.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import numpy as np
import scipy.sparse as sps

class SLIM_BPR(Recommender):
    
    def __init__(self, ICM, URM, URM_train, URM_test, target_playlists):
        
        Recommender.__init__(self,ICM, URM, URM_train, URM_test, target_playlists) 

    def fit(self, URM):
 
        self.recommender = SLIMElasticNetRecommender(URM)
        #self.recommender.fit(l1_penalty=0.1, l2_penalty=0.1, positive_only=True)
        self.recommender.fit(l1_ratio=0.1, positive_only=True, topK = 100)

        matrix_product = np.dot(self.recommender.W_sparse, URM.T)
        self.matrix_product_csc = sps.csc_matrix(matrix_product)
    
        # number of item interactions of the users
        self.corated_items = [len(URM[user].indices) for user \
                in np.arange(URM.shape[0])]
        self.URM_setted = URM        

    def recommend(self, user_id):
        
        # take a single column of product matrix. column_product has size item numbers.
        column_product = self.matrix_product_csc[:,user_id].toarray().squeeze()
             
        # item based score of user vector of items
        item_score = self.item_based_score(column_product, self.corated_items, user_id, self.URM_setted)
        
        submission = np.argsort(item_score)[::-1][:10]
        return submission

    def item_based_score(self, column_product, corated_items, user, URM):
        # all sij are < 1 and weighted by the occurency of the items
        column_norm = np.divide(column_product, corated_items[user])
            
        # put to zero all nan values; it happens only on testing
        column_norm_no_nan = np.where(np.isnan(column_norm), [0], column_norm) 
            
        # put to zero the items score already chosen by the user 
        mask_one_interaction = np.logical_not(URM[user].toarray().squeeze()) 
        column_ratings = np.multiply(column_norm_no_nan, mask_one_interaction)

        return column_ratings
	
