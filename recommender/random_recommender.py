from .recommender import Recommender
import numpy as np

"""
Random Recommender

 For each user/playlist it uniformly performs 
a random choice on the unrated items of that user 

"""

class RandomRecommender(Recommender):
    
    def recommend_all(self, URM_matrix, test_set_size, at):
        submission = []
        random_range = [item for item in range(self.URM.shape[1])]
        
        for playlist in range(test_set_size):
            
            start_pos = self.URM.indptr[playlist]
            end_pos = self.URM.indptr[playlist + 1]

            one_elements = self.URM.indices[start_pos:end_pos]
            zero_elements = list(set(random_range).difference(one_elements))
            
            submission.append([playlist, np.random.choice(zero_elements, size = at, replace = True)])
        return submission

    def recommend(self, playlist):
        
        submission = self.submission[playlist][1]
        return submission 
