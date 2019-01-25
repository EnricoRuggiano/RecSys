from lightfm import LightFM
from .recommender import Recommender
import modules.Compute_Similarity_Python as sim
import modules.item_score as item
import modules.feature_score as feature
import modules.sequential_score as sequential
import modules.duration_score as f_duration
import modules.combine as combine

import time as time
import scipy.sparse as sps
import numpy as np

class HybridRecommender(Recommender):

    
    def __init__(self, ICM, URM, URM_train, URM_test, target_playlists, sequential_list, sequential_list_train, sequential_list_test):
        
        Recommender.__init__(self,ICM, URM, URM_train, URM_test, target_playlists, sequential_list, sequential_list_train, sequential_list_test) 

    def fit(self, URM, sequential_list, artist = True, album = True, duration = False):
        self.URM_setted = URM
        self.ICM_setted = self.ICM
        self.sequential_list_setted = sequential_list

        if(artist):
            self.f = feature.calculate_feature_matrix(0, URM, self.ICM_setted)
        if(album):
            self.a = feature.calculate_feature_matrix(1, URM, self.ICM_setted)
        if(duration):
            self.d = f_duration.duration_all_items(self.ICM_setted)
 
        sim_obj_item = sim.Compute_Similarity_Python(URM, topK=100, shrink= 100, \
                similarity='cosine') 
        self.S_item = sim_obj_item.compute_similarity()

        self.lightfm_rec = LightFM(no_components = 200, learning_schedule = 'adagrad', \
              loss='warp', learning_rate = 0.03)
        self.lightfm_rec.fit(self.URM_setted, epochs=150, num_threads=4)
        

    def recommend_all(self):

        with open(self.output_path, 'w') as f: 
            print("playlist_id,track_ids", end='\n', file=f)
            for user_id in self.target_playlists:
                rank = self.recommend(user_id)
                print(user_id, end=',', file = f)
                print(*rank, sep=' ', end='\n', file = f)


       
    def recommend(self, user_id):

        # lightfm score
        lightfm_score = self.lightfm_rec.predict(user_id, np.arange(self.URM_setted.shape[1]), num_threads=4)
        np.put(lightfm_score, self.URM_setted[user_id].indices, -np.inf)
        lightfm_not_inf = np.where(np.isinf(lightfm_score), np.nan, lightfm_score)
        lightfm_not_nan = np.where(np.isnan(lightfm_not_inf), np.nanmin(lightfm_not_inf), lightfm_not_inf) # assign the min score to already saw items
        lightfm_score = lightfm_not_nan

        # CBF item_based score
        item_score = item.item_score(user_id, self.URM_setted, self.S_item, self.sequential_list_setted)        
        
        # CB score 
        artist_score = feature.feature_score(user_id, 0, self.f, self.URM_setted, self.ICM_setted)
        album_score = feature.feature_score(user_id, 1, self.a, self.URM_setted, self.ICM_setted)
        # duration_score = f_duration.calculate_duration_score(user_id, self.d, self.URM_setted, self.ICM_setted) 

        # first layer of hybrid
        item_weigths = np.array([[1], [0.8]])
        item_boost = combine.combine([item_score, lightfm_score], item_weigths, normalization=True)

        # second layer of hybrid
        arglist_fin =[item_boost, artist_score, album_score]
        weigths_fin = np.array([[0.9], [0.1], [0.1]])
        submission = combine.combine(arglist_fin, weigths_fin, normalization=True)
            
        # top 10 elements 
        score = np.argsort(submission)[::-1][:10]
        return score
