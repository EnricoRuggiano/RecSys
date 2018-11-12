from .recommender import Recommender
import numpy as np

"""
Random Recommender

 For each user/playlist it uniformly performs 
a random choice on the unrated items of that user 

"""

class RandomRecommender(Recommender):

    def __init__(self, ICM, URM, target_playlists):
        self.random_range = [item for item in range(URM.shape[1])]
        Recommender.__init__(self, ICM, URM, target_playlists)

    def recommend_all(self):
        submission = []
        URM_matrix = self.URM.todense()
        for playlist in self.target_playlists:
            zero_element = list(np.argwhere(URM_matrix[playlist] == 0)[:,0])
            submission.append([playlist, np.random.choice(zero_element, size = 10, replace = True)])
        return submission

    def recommend(self, playlist, at):
        
    #    start_pos = self.URM.indptr[playlist]
    #    end_pos = self.URM.indptr[playlist + 1] 

    #    one_elements = self.URM.indices[start_pos:end_pos]
    #    zero_elements = [item for item in self.random_range if item not in one_elements]

        submission = np.random.choice(self.random_range, size = at, replace = True)
        return submission
    
    def execute(self):
        self.submission = self.recommend_all()
        self.submit_solution()
