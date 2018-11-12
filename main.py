import modules.importer as imp
import modules.data_splitter as splitter
import modules.evaluation_function as evaluation
import recommender.random_recommender as rc
import recommender.popularity_recommender as top_pop

if __name__ == '__main__':
    ICM_matrix = imp.create_ICM()
    URM_matrix = imp.create_URM()
    target_playlists = imp.import_target_playlist()
    
    (URM_train, URM_test) = splitter.train_test_holdout(URM_matrix)

    # Random_recommender
    #random_recommender = rc.RandomRecommender(ICM_matrix, URM_train, target_playlists)
    #random_recommender.execute()
    
    top_pop_recommender = top_pop.TopPopRecommender(ICM_matrix, URM_train, target_playlists)
    top_pop_recommender.fit()
    evaluation.evaluate_algorithm(URM_test, top_pop_recommender) 
