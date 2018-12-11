from .recommender import Recommender
import numpy as np

class TopPopRecommender(Recommender):

    def fit(self, URM_matrix):
       
        num_interactions = np.sum(URM_matrix, axis = 0)
        array_interactions = (np.array(num_interactions)).squeeze()
        
        self.top_pop_items = np.flip(np.argsort(array_interactions))
 
    def recommend_all(self, URM_matrix, test_set, at):

        submission = []
        mask = [item for item in range(at)]
        for playlist in test_set:
            
            start_pos = URM_matrix.indptr[playlist]
            end_pos = URM_matrix.indptr[playlist + 1]

            one_elements = URM_matrix.indices[start_pos:end_pos]
            not_ranked_items = np.delete(self.top_pop_items, one_elements)
            top_ten = np.take(not_ranked_items, mask)
            submission.append([playlist, top_ten])
        return submission   

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
   
