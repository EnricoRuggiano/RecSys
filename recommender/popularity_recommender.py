from .recommender import Recommender
import numpy as np

class TopPopRecommender(Recommender):

    def fit(self):
        number_items = self.URM.shape[1]
        item_number_interaction = np.sum(self.URM, axis = 0)
        interaction_item_sorted = np.sort(item_number_interaction)
        
        self.top_items = interaction_item_sorted[-10:]

    def recommend(self, playlist, at):

        submission = self.top_items
        return submission
    
    def recommend_all(self):
        submission = []
        URM_matrix = self.URM.todense()
        for playlist in self.target_playlists:
            submission.append([playlist, self.top_items])
        return submission
    
    def execute(self):
        self.submission = self.recommend_all()
        self.submit_solution()

