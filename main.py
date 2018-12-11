from recommender.default.Base.Evaluation.Evaluator import SequentialEvaluator

import modules.importer as imp
import modules.data_splitter as splitter
import modules.evaluation_function as evaluation
import recommender.random_recommender as rc
import recommender.popularity_recommender as top_pop
import recommender.content_based as cb
import recommender.user_based as cub
import recommender.item_based as cib
import recommender.hybrid_item_user as hiu
import recommender.item_based_duration as icd
import recommender.pure_SVD as svd
import recommender.slim_bpr as slim_bpr
import recommender.hybrid_item_svd as isv
import recommender.funk as seb

if __name__ == '__main__':
    ICM_matrix = imp.create_ICM()
    URM_matrix = imp.create_URM()
    target_playlists = imp.import_target_playlist()
    
    (URM_train, URM_test) = splitter.train_test_holdout(URM_matrix, train_perc = 0.8)
    #evaluator = SequentialEvaluator(URM_test, cutoff_list=[10])
    # Random_recommender
    #random_recommender = rc.RandomRecommender(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
    #random_recommender.execute() 
    #random_recommender.test()
    #evaluation.evaluate_algorithm(URM_test, random_recommender) 
    
#    tpop_recommender = top_pop.TopPopRecommender(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
#    tpop_recommender.execute()
#    tpop_recommender.test()
#    evaluation.evaluate_algorithm(URM_test, tpop_recommender)
#   cb_recommender = cb.CBRecommender(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
#    cb_recommender.execute()
#    cb_recommender.test()
#    evaluation.evaluate_algorithm(URM_test, cb_recommender)
#    cub_recommender = cub.CFUB_recommender(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
#    cub_recommender.execute()
#    cub_recommender.test()
#    evaluation.evaluate_algorithm(URM_test, cub_recommender)
#    cib_recommender = cib.CFIB_recommender(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
#    cib_recommender.execute()
#    cib_recommender.test()
#    evaluation.evaluate_algorithm(URM_test, cib_recommender)

#    hiu_recommender = hiu.CFHY_recommender(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
#    hiu_recommender.execute()
#    hiu_recommender.test()
#    evaluation.evaluate_algorithm(URM_test, hiu_recommender)

#    icd_recommender = icd.CFIBD_recommender(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
#    icd_recommender.execute()
#    icd_recommender.test()
#    evaluation.evaluate_algorithm(URM_test, icd_recommender)

    slim_bpr_recommender = slim_bpr.SLIM_BPR(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
    #slim_bpr_recommender.execute()
    slim_bpr_recommender.test()
    evaluation.evaluate_algorithm(URM_test, slim_bpr_recommender)
#    slim_bpr_recommender.fit(URM_matrix)
#    slim_bpr_recommender.recommend_all()

#    slim_bpr_elastic_recommender = seb.MF_BPR(ICM_matrix, URM_matrix, URM_train, URM_test, target_playlists)
    ##slim_bpr_recommender.execute()
#    slim_bpr_elastic_recommender.test()
    #evaluator.evaluateRecommender(slim_bpr_elastic_recommender)
    #evaluation.evaluate_algorithm(URM_test, slim_bpr_elastic_recommender)
    #slim_bpr_recommender.fit(URM_matrix)
    #slim_bpr_recommender.recommend_all()
